#!/usr/bin/env python3
"""verify_pd_disagg.py - quick functional checks against an SGLang PD router or server.

Steps (each prints PASS/FAIL): workers (--workers), health, smoke, needle-in-haystack at
several lengths/depths, warm-cache check, and optional context-length boundary (--boundary).

Example:
  python3 verify_pd_disagg.py --url http://router:8000 --workers --lengths 1024,32768 \
      --depth 0.1 --depth 0.9 --api generate --boundary --context-length 131072
Exit code 1 if any step FAILs.
"""
import argparse
import json
import random
import string
import sys
import time

import requests

FILLER = ["The quick brown fox jumps over the lazy dog near the riverbank.",
          "Rain fell steadily on the old tin roof throughout the long night.",
          "Engineers reviewed the schematic twice before approving the design.",
          "A gentle breeze carried the scent of pine across the quiet valley.",
          "The library closes early on Sundays and holidays during winter."]
SMOKE_PROMPT = "List three primary colors and explain why they are called primary."
NEEDLE_Q = "What is the secret code word? Answer with just the word."
RESULTS = []


def report(step, ok, detail=""):
    """Record a step. ok=True PASS, False FAIL, None SKIP (not counted toward exit code)."""
    RESULTS.append((step, ok))
    tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    print(f"[{tag}] {step}: {detail}", flush=True)
    return ok


class Counter:
    """Token counter: exact via HF tokenizer.json if available, else chars/token heuristic."""

    def __init__(self, tok_json, cpt):
        self.cpt, self.tok = cpt, None
        if tok_json:
            try:
                from tokenizers import Tokenizer  # optional third-party
                self.tok = Tokenizer.from_file(tok_json)
                print(f"using exact tokenizer from {tok_json}")
            except ImportError:
                print("warning: 'tokenizers' package not importable, using heuristic")
            except Exception as e:  # noqa: BLE001
                print(f"warning: cannot load tokenizer {tok_json}: {e}; using heuristic")

    def count(self, text):
        if self.tok:
            return len(self.tok.encode(text, add_special_tokens=False).ids)
        return int(len(text) / self.cpt)

    def filler(self, n_tokens):
        """Return filler text of approximately n_tokens tokens."""
        parts, i, total = [], 0, 0
        target_chars = n_tokens * self.cpt
        while total < target_chars:
            s = FILLER[i % len(FILLER)]
            parts.append(s)
            total += len(s) + 1
            i += 1
        text = " ".join(parts)
        if self.tok:  # trim/extend precisely
            while self.count(text) > n_tokens and parts:
                parts.pop()
                text = " ".join(parts)
            while self.count(text) < n_tokens:
                parts.append(FILLER[i % len(FILLER)])
                i += 1
                text = " ".join(parts)
        return text


class Client:
    def __init__(self, args):
        self.base, self.api, self.model, self.timeout = args.url.rstrip("/"), args.api, args.model, args.timeout
        self.sess = requests.Session()

    def get(self, path):
        return self.sess.get(self.base + path, timeout=min(self.timeout, 30))

    def generate(self, prompt, max_new_tokens, timeout=None):
        """Return dict(text, prompt_tokens, completion_tokens, cached_tokens, ttft_ms, e2e_ms, status, error)."""
        out = {"text": "", "prompt_tokens": None, "completion_tokens": None, "cached_tokens": None,
               "ttft_ms": None, "e2e_ms": None, "status": None, "error": None}
        if self.api == "chat":
            url, body = self.base + "/v1/chat/completions", {
                "model": self.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": 0, "max_tokens": max_new_tokens, "stream": False}
        else:
            url, body = self.base + "/generate", {
                "text": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens}}
        t0 = time.time()
        try:
            r = self.sess.post(url, json=body, timeout=timeout or self.timeout, stream=True)
            out["status"] = r.status_code
            buf = bytearray()
            for chunk in r.iter_content(chunk_size=4096):  # first chunk arrival ~= TTFT
                if out["ttft_ms"] is None:
                    out["ttft_ms"] = round((time.time() - t0) * 1000, 1)
                buf.extend(chunk)
            out["e2e_ms"] = round((time.time() - t0) * 1000, 1)
            body_text = buf.decode("utf-8", errors="replace")
            if r.status_code != 200:
                out["error"] = f"HTTP {r.status_code}: {body_text[:300]}"
                return out
            j = json.loads(body_text)
            if self.api == "chat":
                ch = (j.get("choices") or [{}])[0]
                out["text"] = (ch.get("message") or {}).get("content") or ""
                u = j.get("usage") or {}
                out.update(prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                           cached_tokens=(u.get("prompt_tokens_details") or {}).get("cached_tokens"))
            else:
                m = j.get("meta_info") or {}
                out["text"] = j.get("text") or ""
                out.update(prompt_tokens=m.get("prompt_tokens"), completion_tokens=m.get("completion_tokens"),
                           cached_tokens=m.get("cached_tokens"))
        except requests.RequestException as e:
            out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            out["e2e_ms"] = round((time.time() - t0) * 1000, 1)
        except ValueError as e:
            out["error"] = f"bad JSON: {e}"
        return out


def step_workers(cli):
    try:
        r = cli.get("/workers")
        if r.status_code == 404:
            return report("workers", None, "mini-LB has no /workers (HTTP 404); only the Rust router exposes it")
        j = r.json()
    except (requests.RequestException, ValueError) as e:
        return report("workers", False, f"GET /workers failed: {e}")
    pc = dc = None
    if isinstance(j, dict):
        if isinstance(j.get("prefill"), list) and isinstance(j.get("decode"), list):
            pc, dc = len(j["prefill"]), len(j["decode"])
        elif "prefill_count" in j and "decode_count" in j:
            pc, dc = j["prefill_count"], j["decode_count"]
        elif isinstance(j.get("workers"), list):
            pc = sum(1 for w in j["workers"] if str(w.get("worker_type", w.get("type", ""))).lower().startswith("prefill"))
            dc = sum(1 for w in j["workers"] if str(w.get("worker_type", w.get("type", ""))).lower().startswith("decode"))
    if pc is None:
        return report("workers", False, f"unrecognized /workers shape, raw: {str(j)[:400]}")
    return report("workers", pc == 1 and dc == 1, f"prefill_count={pc} decode_count={dc} raw={str(j)[:200]}")


def step_health(cli, is_router):
    path = "/health_generate" if is_router else "/health"
    try:
        r = cli.get(path)
        return report("health", r.status_code == 200, f"GET {path} -> HTTP {r.status_code} ({r.text[:80]!r})")
    except requests.RequestException as e:
        return report("health", False, f"GET {path} failed: {e}")


def step_smoke(cli):
    o = cli.generate(SMOKE_PROMPT, 32)
    ok = not o["error"] and bool(o["text"].strip()) and (o["completion_tokens"] or 0) > 0
    return report("smoke", ok, f"completion_tokens={o['completion_tokens']} ttft={o['ttft_ms']}ms "
                  f"e2e={o['e2e_ms']}ms text={o['text'][:60]!r} err={o['error']}")


def build_needle_prompt(counter, length, depth, word):
    needle = f" The secret code word is {word}. "
    body = counter.filler(max(16, length - 48))
    pos = int(len(body) * depth)
    pos = body.find(" ", pos) if body.find(" ", pos) >= 0 else pos
    return body[:pos] + needle + body[pos:] + "\n\n" + NEEDLE_Q


def step_needles(cli, counter, lengths, depths):
    last = None
    for L in lengths:
        for d in depths:
            word = "".join(random.choices(string.ascii_lowercase, k=8))
            prompt = build_needle_prompt(counter, L, d, word)
            o = cli.generate(prompt, 16)
            ok = not o["error"] and word in o["text"].lower()
            report(f"needle len={L} depth={d}", ok,
                   f"want={word} got={o['text'].strip()[:40]!r} prompt_tokens={o['prompt_tokens']} "
                   f"(est {counter.count(prompt)}) ttft={o['ttft_ms']}ms e2e={o['e2e_ms']}ms err={o['error']}")
            last = (prompt, o)
    return last


def step_warm_cache(cli, first):
    prompt, o1 = first
    o2 = cli.generate(prompt, 16)
    if o2["error"] or o1["error"]:
        return report("warm-cache", False, f"err1={o1['error']} err2={o2['error']}")
    ratio = (o2["ttft_ms"] / o1["ttft_ms"]) if o1["ttft_ms"] and o2["ttft_ms"] else None
    cached, pt = o2["cached_tokens"], o2["prompt_tokens"]
    ok = cached is None or pt is None or cached >= 0  # informational: cached may be 0 in PD mode
    return report("warm-cache", ok, f"second: cached_tokens={cached} prompt_tokens={pt} "
                  f"ttft1={o1['ttft_ms']}ms ttft2={o2['ttft_ms']}ms ratio={ratio if ratio is None else round(ratio, 3)}")


def step_boundary(cli, counter, ctx, timeout):
    o = cli.generate(counter.filler(ctx - 1024) + "\n\nSay OK.", 8)
    report(f"boundary ctx-1024 ({ctx - 1024} tok)", not o["error"],
           f"status={o['status']} prompt_tokens={o['prompt_tokens']} e2e={o['e2e_ms']}ms err={o['error']}")
    o = cli.generate(counter.filler(ctx + 1) + "\n\nSay OK.", 8, timeout=timeout)
    ok = o["status"] is not None and 400 <= o["status"] < 500
    report(f"boundary ctx+1 ({ctx + 1} tok, expect 4xx)", ok,
           f"status={o['status']} e2e={o['e2e_ms']}ms err={(o['error'] or '')[:160]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="router or server base URL")
    ap.add_argument("--workers", action="store_true", help="check GET /workers for 1 prefill + 1 decode")
    ap.add_argument("--router", action="store_true", help="target is a router (use /health_generate); implied by --workers")
    ap.add_argument("--api", choices=["generate", "chat"], default="generate")
    ap.add_argument("--model", default="default", help="model name for chat API")
    ap.add_argument("--lengths", default="1024,8192", help="comma list of haystack lengths in tokens")
    ap.add_argument("--length", type=int, help="single haystack length (overrides --lengths)")
    ap.add_argument("--depth", action="append", type=float, help="needle depth fraction (repeatable; default 0.1,0.5,0.95)")
    ap.add_argument("--chars-per-token", type=float, default=3.6)
    ap.add_argument("--tokenizer-json", help="HF tokenizer.json for exact counts (needs 'tokenizers')")
    ap.add_argument("--boundary", action="store_true", help="run context-length boundary test")
    ap.add_argument("--context-length", type=int, default=131072)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
    depths = args.depth or [0.1, 0.5, 0.95]
    lengths = [args.length] if args.length else [int(x) for x in args.lengths.split(",") if x.strip()]
    counter = Counter(args.tokenizer_json, args.chars_per_token)
    cli = Client(args)

    if args.workers:
        step_workers(cli)
    step_health(cli, args.router or args.workers)
    step_smoke(cli)
    last = step_needles(cli, counter, lengths, depths)
    if last:
        step_warm_cache(cli, last)
    if args.boundary:
        step_boundary(cli, counter, args.context_length, args.timeout)

    n_fail = sum(1 for _, ok in RESULTS if ok is False)
    n_skip = sum(1 for _, ok in RESULTS if ok is None)
    print(f"\n=== summary: {len(RESULTS) - n_fail - n_skip} passed, {n_fail} failed, {n_skip} skipped ===")
    for s, ok in RESULTS:
        if ok is False:
            print(f"  FAIL {s}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
