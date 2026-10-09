#!/usr/bin/env python3
"""mm_request.py - build (and optionally send) a Qwen3-VL multimodal chat request.

Example:
  python3 mm_request.py --url http://router:8000 --model qwen3-vl --num-images 4 \
      --resolution 896x896 --target-tokens 16384 --needle-word pelican --stream --out resp.json
  python3 mm_request.py --num-images 2 --dry-run          # payload size + token estimate only

Images: synthetic (default; unique colored shape + index text, verifiable), noise, or file
(--image-path, repeatable).  Requires Pillow for synthetic/noise images; numpy is optional.
Token estimate per image: (W/28)*(H/28)/4 (2x2 patch merge); text at 3.6 chars/token.
"""
import argparse
import base64
import io
import json
import os
import sys
import time

import requests

FILLER = ["The museum catalog lists every painting by year of acquisition.",
          "Trains on the northern line run every twelve minutes after noon.",
          "Granite weathers slowly compared to the softer sandstone nearby.",
          "The recipe calls for two cups of flour and a pinch of salt.",
          "Volunteers planted three hundred saplings along the creek bed."]
SHAPES = ["circle", "square", "triangle", "diamond", "star"]
COLORS = [("red", (220, 40, 40)), ("green", (40, 170, 60)), ("blue", (40, 80, 220)),
          ("orange", (240, 150, 30)), ("purple", (140, 50, 180)), ("teal", (20, 160, 160)),
          ("yellow", (230, 210, 40)), ("pink", (240, 100, 170))]
CHARS_PER_TOKEN = 3.6


def need_pil():
    try:
        from PIL import Image, ImageDraw  # noqa: F401
        return Image, ImageDraw
    except ImportError:
        sys.exit("error: Pillow is required for --image-kind synthetic|noise "
                 "(pip install pillow), or use --image-kind file --image-path ...")


def synthetic_image(idx, w, h):
    """Draw a unique colored shape and the image index; return (png_bytes, description)."""
    Image, ImageDraw = need_pil()
    cname, rgb = COLORS[idx % len(COLORS)]
    shape = SHAPES[idx % len(SHAPES)]
    img = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(img)
    cx, cy, r = w // 2, h // 2, min(w, h) // 3
    if shape == "circle":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)
    elif shape == "square":
        d.rectangle([cx - r, cy - r, cx + r, cy + r], fill=rgb)
    elif shape == "triangle":
        d.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=rgb)
    elif shape == "diamond":
        d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=rgb)
    else:
        import math
        pts = []
        for k in range(10):
            rr = r if k % 2 == 0 else r * 0.45
            a = -math.pi / 2 + k * math.pi / 5
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        d.polygon(pts, fill=rgb)
    label = f"IMAGE {idx + 1}"
    # draw label large: scale a default bitmap font by drawing it repeatedly in a block
    scale = max(2, min(w, h) // 120)
    small = Image.new("RGB", (len(label) * 6 + 4, 12), (245, 245, 245))
    ImageDraw.Draw(small).text((2, 1), label, fill=(0, 0, 0))
    big = small.resize((small.width * scale, small.height * scale), Image.NEAREST)
    img.paste(big, (8, 8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), f"{cname} {shape} labeled '{label}'"


def noise_image(w, h):
    Image, _ = need_pil()
    try:
        import numpy as np
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        img = Image.fromarray(arr, "RGB")
    except ImportError:
        img = Image.frombytes("RGB", (w, h), os.urandom(w * h * 3))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), "random noise"


def file_image(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        sys.exit(f"error: cannot read --image-path {path}: {e}")
    ext = os.path.splitext(path)[1].lower().lstrip(".") or "png"
    return data, f"file {os.path.basename(path)}", {"jpg": "jpeg"}.get(ext, ext)


def filler_text(n_tokens):
    out, i = [], 0
    while sum(len(s) + 1 for s in out) < n_tokens * CHARS_PER_TOKEN:
        out.append(FILLER[i % len(FILLER)])
        i += 1
    return " ".join(out)


def est_text_tokens(s):
    return int(len(s) / CHARS_PER_TOKEN)


def build_payload(args):
    w, h = args.w, args.h
    per_img = args.tokens_per_image if args.tokens_per_image else int((w / 28) * (h / 28) / 4)
    images, descs = [], []
    for i in range(args.num_images):
        if args.image_kind == "synthetic":
            data, desc = synthetic_image(i, w, h)
            mime = "png"
        elif args.image_kind == "noise":
            data, desc = noise_image(w, h)
            mime = "png"
        else:
            if not args.image_path:
                sys.exit("error: --image-kind file requires --image-path")
            data, desc, mime = file_image(args.image_path[i % len(args.image_path)])
        images.append({"type": "image_url", "image_url": {
            "url": f"data:image/{mime};base64," + base64.b64encode(data).decode()}})
        descs.append(desc)

    needle = f" The secret word is {args.needle_word}. " if args.needle_word else ""
    fixed_tokens = per_img * args.num_images + est_text_tokens(args.question + needle) + 32
    pad_tokens = max(0, (args.target_tokens or 0) - fixed_tokens)
    pre_tokens = min(pad_tokens, args.image_straddle_chunk) if args.image_straddle_chunk else 0
    pre = filler_text(pre_tokens) if pre_tokens else ""
    post = filler_text(pad_tokens - pre_tokens) if pad_tokens - pre_tokens > 0 else ""
    if needle:
        cut = int(len(post) * args.needle_depth)
        sp = post.find(" ", cut)
        cut = sp if sp >= 0 else cut
        post = post[:cut] + needle + post[cut:]

    content = []
    if pre:
        content.append({"type": "text", "text": pre})
    content.extend(images)
    if post:
        content.append({"type": "text", "text": post})
    content.append({"type": "text", "text": args.question})
    payload = {"model": args.model, "messages": [{"role": "user", "content": content}],
               **({"priority": args.priority} if args.priority is not None else {}),
               "max_tokens": args.max_tokens, "temperature": args.temperature, "stream": args.stream}
    if args.ignore_eos:
        payload["ignore_eos"] = True
    if args.stream:
        payload["stream_options"] = {"include_usage": True}
    est = per_img * args.num_images + est_text_tokens(pre + post + args.question)
    return payload, est, per_img, descs


def send(args, payload):
    url = args.url.rstrip("/") + "/v1/chat/completions"
    t0 = time.time()
    res = {"text": "", "usage": None, "ttft_ms": None, "e2e_ms": None, "status": None, "error": None}
    try:
        r = requests.post(url, json=payload, timeout=args.timeout, stream=args.stream)
        res["status"] = r.status_code
        if r.status_code != 200:
            res["error"] = f"HTTP {r.status_code}: {r.text[:500]}"
            res["e2e_ms"] = round((time.time() - t0) * 1000, 1)
            return res
        if args.stream:
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    res["usage"] = chunk["usage"]
                ch = chunk.get("choices") or []
                piece = (ch[0].get("delta") or {}).get("content") if ch else None
                if piece:
                    if res["ttft_ms"] is None:
                        res["ttft_ms"] = round((time.time() - t0) * 1000, 1)
                    res["text"] += piece
                    if args.verbose:
                        print(piece, end="", flush=True)
            if args.verbose:
                print()
        else:
            res["ttft_ms"] = round((time.time() - t0) * 1000, 1)
            j = r.json()
            ch = (j.get("choices") or [{}])[0]
            res["text"] = (ch.get("message") or {}).get("content") or ""
            res["usage"] = j.get("usage")
            res["raw"] = j
        res["e2e_ms"] = round((time.time() - t0) * 1000, 1)
    except requests.RequestException as e:
        res["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        res["e2e_ms"] = round((time.time() - t0) * 1000, 1)
    except ValueError as e:
        res["error"] = f"bad response JSON: {e}"
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000", help="router/server base URL")
    ap.add_argument("--model", default="default")
    ap.add_argument("--num-images", type=int, default=1)
    ap.add_argument("--resolution", default="448x448", help="WxH of generated images")
    ap.add_argument("--target-tokens", type=int, default=None, help="pad text so prompt is ~T tokens")
    ap.add_argument("--tokens-per-image", type=int, default=None, help="override per-image token estimate")
    ap.add_argument("--question", default="Describe each image briefly and then state the secret word.")
    ap.add_argument("--needle-word", default=None)
    ap.add_argument("--needle-depth", type=float, default=0.5)
    ap.add_argument("--image-kind", choices=["synthetic", "noise", "file"], default="synthetic")
    ap.add_argument("--image-path", action="append", help="image file (repeatable, for --image-kind file)")
    ap.add_argument("--image-straddle-chunk", type=int, default=None, metavar="CH",
                    help="place first image after ~CH filler tokens (needs --target-tokens > CH)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--priority", type=int, default=None, help="request priority (needs --enable-priority-scheduling)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print payload size/token estimate, do not send")
    ap.add_argument("--out", help="save response JSON here")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--verbose", action="store_true", help="echo streamed text")
    args = ap.parse_args()
    try:
        args.w, args.h = (int(x) for x in args.resolution.lower().split("x"))
    except ValueError:
        sys.exit(f"error: --resolution must be WxH, got {args.resolution!r}")
    if args.image_straddle_chunk and not args.target_tokens:
        sys.exit("error: --image-straddle-chunk requires --target-tokens")

    payload, est, per_img, descs = build_payload(args)
    size = len(json.dumps(payload))
    print(f"payload: {size / 1024:.1f} KiB, {args.num_images} image(s) {args.w}x{args.h} "
          f"(~{per_img} tok each), est prompt tokens ~{est}, content parts={len(payload['messages'][0]['content'])}")
    for i, d in enumerate(descs):
        print(f"  image {i + 1}: {d}")
    if args.dry_run:
        return
    res = send(args, payload)
    if res["error"]:
        print(f"ERROR {res['error']}")
    u = res.get("usage") or {}
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", u.get("cached_tokens"))
    print(f"status={res['status']} prompt_tokens={u.get('prompt_tokens')} "
          f"completion_tokens={u.get('completion_tokens')} cached_tokens={cached} "
          f"ttft={res['ttft_ms']}ms e2e={res['e2e_ms']}ms")
    print(f"text: {res['text'][:500]!r}")
    if args.needle_word:
        print(f"needle {'FOUND' if args.needle_word.lower() in res['text'].lower() else 'MISSING'}: {args.needle_word}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({k: v for k, v in res.items()} | {"image_descriptions": descs, "est_prompt_tokens": est},
                      f, indent=1)
        print(f"saved -> {args.out}")
    sys.exit(1 if res["error"] else 0)


if __name__ == "__main__":
    main()
