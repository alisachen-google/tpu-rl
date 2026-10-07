#!/usr/bin/env python3
"""golden_diff.py - generate greedy outputs from an SGLang server/router and diff two runs.

  generate: python3 golden_diff.py generate --prompts prompts.jsonl --url http://h:p \
              --out outputs.jsonl [--max-new-tokens 1024] [--ignore-eos] [--return-logprob] \
              [--concurrency 4] [--stream] [--timeout 7200] [--keep-itl] [--model NAME]
  diff:     python3 golden_diff.py diff --a golden.jsonl --b pd.jsonl --report report.json \
              [--expect-tokens N] [--exact-prefix-tokens 256] [--min-agreement 0.95]

prompts.jsonl lines: {"id":..., "text":..., "max_new_tokens": opt} or {"id":..., "messages":[...]}.
No retries are performed: failures are recorded as-is.  diff exits 1 if any id fails a gate.
"""
import argparse
import json
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests


# ----------------------------------------------------------------------------- generate
def percentile(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(p / 100 * len(s)) - 1))]


def extract_token_ids(meta):
    """Pull output token ids from meta_info if present (output_ids or output_token_logprobs)."""
    if not isinstance(meta, dict):
        return None
    if isinstance(meta.get("output_ids"), list):
        return meta["output_ids"]
    lp = meta.get("output_token_logprobs")
    if isinstance(lp, list) and lp and isinstance(lp[0], (list, tuple)) and len(lp[0]) >= 2:
        return [int(t[1]) for t in lp if t[1] is not None]
    return None


def first_logprob(meta):
    lp = (meta or {}).get("output_token_logprobs")
    if isinstance(lp, list) and lp and isinstance(lp[0], (list, tuple)):
        return lp[0][0]
    return None


def read_sse(resp, t0, rec, chat):
    """Consume an SSE stream; fill rec text/ttft/itl; return final meta (generate) or usage (chat)."""
    text, last_t, meta, last_chunk = "", None, {}, None
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        payload = raw[5:].strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        now = time.time()
        if chat:
            ch = chunk.get("choices") or [{}]
            piece = (ch[0].get("delta") or {}).get("content") or ""
            if ch[0].get("finish_reason"):
                meta["finish_reason"] = ch[0]["finish_reason"]
            if chunk.get("usage"):
                meta["usage"] = chunk["usage"]
            if piece:
                text += piece
        else:
            piece = chunk.get("text", "")
            text = piece if len(piece) >= len(text) else text + piece  # sglang sends cumulative text
            meta = chunk.get("meta_info", meta) or meta
            last_chunk = chunk
        if piece or not chat:
            if rec["ttft_ms"] is None:
                rec["ttft_ms"] = round((now - t0) * 1000, 2)
            elif last_t is not None:
                rec["_itl"].append((now - last_t) * 1000)
            last_t = now
    rec["text"] = text
    return meta, last_chunk


def run_one(prompt, args, sess):
    rec = {"id": prompt.get("id"), "text": "", "completion_tokens": None, "finish_reason": None,
           "cached_tokens": None, "prompt_tokens": None, "ttft_ms": None, "e2e_ms": None,
           "itl_max_ms": None, "itl_p99_ms": None, "first_token_logprob": None,
           "output_token_ids": None, "error": None, "_itl": []}
    chat = "messages" in prompt
    mnt = int(prompt.get("max_new_tokens", args.max_new_tokens))
    base = args.url.rstrip("/")
    if chat:
        url = base + "/v1/chat/completions"
        body = {"model": args.model, "messages": prompt["messages"], "temperature": 0,
                "max_tokens": mnt, "stream": args.stream}
        if args.ignore_eos:
            body["ignore_eos"] = True
        if args.stream:
            body["stream_options"] = {"include_usage": True}
    else:
        if "text" not in prompt:
            rec["error"] = "prompt has neither 'text' nor 'messages'"
            return rec
        url = base + "/generate"
        body = {"text": prompt["text"], "rid": str(prompt.get("id")), "stream": args.stream,
                "sampling_params": {"temperature": 0, "max_new_tokens": mnt,
                                    "ignore_eos": bool(args.ignore_eos)}}
        if args.return_logprob:
            body.update({"return_logprob": True, "top_logprobs_num": 1, "logprob_start_len": -1})
    t0 = time.time()
    try:
        resp = sess.post(url, json=body, timeout=args.timeout, stream=args.stream)
        if resp.status_code != 200:
            rec["error"] = f"HTTP {resp.status_code}: {resp.text[:500]}"
            rec["e2e_ms"] = round((time.time() - t0) * 1000, 2)
            return rec
        if args.stream:
            meta, _ = read_sse(resp, t0, rec, chat)
            if chat:
                usage = meta.get("usage") or {}
                meta = {"completion_tokens": usage.get("completion_tokens"),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                        "finish_reason": {"type": meta.get("finish_reason")}}
        else:
            rec["ttft_ms"] = round((time.time() - t0) * 1000, 2)  # first byte ~= whole body
            j = resp.json()
            if chat:
                ch = (j.get("choices") or [{}])[0]
                rec["text"] = (ch.get("message") or {}).get("content") or ""
                usage = j.get("usage") or {}
                meta = {"completion_tokens": usage.get("completion_tokens"),
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                        "finish_reason": {"type": ch.get("finish_reason")}}
            else:
                rec["text"] = j.get("text", "")
                meta = j.get("meta_info") or {}
        rec["e2e_ms"] = round((time.time() - t0) * 1000, 2)
        rec["completion_tokens"] = meta.get("completion_tokens")
        rec["prompt_tokens"] = meta.get("prompt_tokens")
        rec["cached_tokens"] = meta.get("cached_tokens")
        fr = meta.get("finish_reason")
        rec["finish_reason"] = fr.get("type") if isinstance(fr, dict) else fr
        rec["first_token_logprob"] = first_logprob(meta)
        rec["output_token_ids"] = extract_token_ids(meta)
        if meta.get("e2e_latency") is not None:
            rec["server_e2e_ms"] = round(float(meta["e2e_latency"]) * 1000, 2)
    except requests.RequestException as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:500]}"
        rec["e2e_ms"] = round((time.time() - t0) * 1000, 2)
    except (ValueError, KeyError, TypeError) as e:
        rec["error"] = f"bad response: {type(e).__name__}: {str(e)[:500]}"
    itl = rec.pop("_itl")
    if itl:
        rec["itl_max_ms"] = round(max(itl), 2)
        rec["itl_p99_ms"] = round(percentile(itl, 99), 2)
        if args.keep_itl:
            rec["itl_ms"] = [round(x, 2) for x in itl]
    return rec


def cmd_generate(args):
    prompts = []
    with open(args.prompts) as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                p = json.loads(line)
            except json.JSONDecodeError as e:
                sys.exit(f"error: {args.prompts}:{ln}: invalid JSON: {e}")
            p.setdefault("id", f"line{ln}")
            prompts.append(p)
    if not prompts:
        sys.exit("error: no prompts found")
    print(f"{len(prompts)} prompts -> {args.url} (concurrency={args.concurrency}, stream={args.stream})")
    sess = requests.Session()
    n_err = 0
    with open(args.out, "w") as out, ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for rec in ex.map(lambda p: run_one(p, args, sess), prompts):
            out.write(json.dumps(rec) + "\n")
            out.flush()
            n_err += bool(rec["error"])
            status = f"ERROR {rec['error'][:120]}" if rec["error"] else (
                f"tokens={rec['completion_tokens']} finish={rec['finish_reason']} cached={rec['cached_tokens']} "
                f"ttft={rec['ttft_ms']}ms e2e={rec['e2e_ms']}ms")
            print(f"[{rec['id']}] {status}", flush=True)
    print(f"done: {len(prompts)} prompts, {n_err} errors -> {args.out}")
    return 1 if n_err else 0


# ----------------------------------------------------------------------------- diff
def load_outputs(path):
    recs = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                recs[str(r.get("id"))] = r
    return recs


def common_prefix(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def max_4gram_repeat(text):
    words = text.split()
    counts = {}
    for i in range(len(words) - 3):
        g = tuple(words[i:i + 4])
        counts[g] = counts.get(g, 0) + 1
    return max(counts.values()) if counts else 0


def structural(rec, args):
    """Return list of structural problems for one side."""
    probs = []
    if rec.get("error"):
        probs.append(f"error:{rec['error'][:60]}")
    if not (rec.get("text") or "").strip():
        probs.append("empty_output")
    if args.expect_tokens is not None and rec.get("completion_tokens") != args.expect_tokens:
        probs.append(f"completion_tokens={rec.get('completion_tokens')}!={args.expect_tokens}")
    reps = max_4gram_repeat(rec.get("text") or "")
    if reps > args.max_4gram_repeats:
        probs.append(f"degenerate_4gram_x{reps}")
    lp = rec.get("first_token_logprob")
    if isinstance(lp, float) and (math.isnan(lp) or math.isinf(lp)):
        probs.append("nan_inf_logprob")
    if isinstance(lp, str) and re.search(r"nan|inf", lp, re.I):
        probs.append("nan_inf_logprob")
    return probs


def cmd_diff(args):
    A, B = load_outputs(args.a), load_outputs(args.b)
    ids = sorted(set(A) | set(B), key=lambda s: (len(s), s))
    rows, n_fail = [], 0
    hdr = f"{'id':<14}{'ident':<6}{'cpfx':>7}{'tpfx':>7}{'agree':>7}{'dlp':>9}  {'gate':<5} notes"
    print(hdr)
    print("-" * len(hdr))
    for i in ids:
        a, b = A.get(i), B.get(i)
        row = {"id": i, "gate_pass": True, "notes": []}
        if a is None or b is None:
            row.update({"gate_pass": False, "notes": [f"missing in {'a' if a is None else 'b'}"]})
        else:
            ta, tb = a.get("text") or "", b.get("text") or ""
            row["identical"] = ta == tb
            row["char_prefix"] = cp = common_prefix(ta, tb)
            ia, ib = a.get("output_token_ids"), b.get("output_token_ids")
            have_tok = isinstance(ia, list) and isinstance(ib, list) and ia and ib
            if have_tok:
                row["token_prefix"] = tp = common_prefix(ia, ib)
                m = min(len(ia), len(ib))
                row["token_agreement"] = agree = sum(x == y for x, y in zip(ia, ib)) / m
                row["first_divergence_token"] = tp if tp < m or len(ia) != len(ib) else None
            else:
                tp = None
                m = min(len(ta), len(tb))
                agree = sum(x == y for x, y in zip(ta, tb)) / m if m else (1.0 if ta == tb else 0.0)
                row["char_agreement"] = agree
            if cp < max(len(ta), len(tb)):
                row["first_divergence_char"] = cp
                row["context_a"] = ta[max(0, cp - 20):cp + 20]
                row["context_b"] = tb[max(0, cp - 20):cp + 20]
            la, lb = a.get("first_token_logprob"), b.get("first_token_logprob")
            if isinstance(la, (int, float)) and isinstance(lb, (int, float)):
                row["first_token_logprob_absdiff"] = abs(la - lb)
            row["structural_a"], row["structural_b"] = structural(a, args), structural(b, args)
            # gates
            if have_tok:
                need = min(args.exact_prefix_tokens, len(ia), len(ib))
                if tp < need:
                    row["notes"].append(f"token prefix {tp}<{need}")
            else:
                need = min(1000, len(ta), len(tb))
                if cp < need:
                    row["notes"].append(f"char prefix {cp}<{need}")
            if agree < args.min_agreement:
                row["notes"].append(f"agreement {agree:.3f}<{args.min_agreement}")
            if row["structural_a"] or row["structural_b"]:
                row["notes"].append("structural:" + ",".join(row["structural_a"] + row["structural_b"]))
            row["gate_pass"] = not row["notes"]
        n_fail += not row["gate_pass"]
        rows.append(row)
        dlp = row.get("first_token_logprob_absdiff")
        print(f"{i:<14}{str(row.get('identical', '-')):<6}{str(row.get('char_prefix', '-')):>7}"
              f"{str(row.get('token_prefix', '-')):>7}"
              f"{(row.get('token_agreement', row.get('char_agreement')) or 0):>7.3f}"
              f"{(f'{dlp:.4f}' if dlp is not None else '-'):>9}  "
              f"{'PASS' if row['gate_pass'] else 'FAIL':<5} {'; '.join(row['notes'])}")
        if row.get("context_a") is not None:
            print(f"{'':14}  a@{row['first_divergence_char']}: {row['context_a']!r}\n{'':14}  b: {row['context_b']!r}")
    summary = {"a": args.a, "b": args.b, "ids": len(ids), "failed": n_fail,
               "identical": sum(1 for r in rows if r.get("identical")), "gates": {
                   "exact_prefix_tokens": args.exact_prefix_tokens, "min_agreement": args.min_agreement,
                   "expect_tokens": args.expect_tokens, "max_4gram_repeats": args.max_4gram_repeats},
               "rows": rows}
    with open(args.report, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"\n{len(ids)} ids, {summary['identical']} identical, {n_fail} failed gates -> {args.report}")
    return 1 if n_fail else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="run prompts against a server/router")
    g.add_argument("--prompts", required=True)
    g.add_argument("--url", required=True, help="http://host:port of server or router")
    g.add_argument("--out", default="outputs.jsonl")
    g.add_argument("--max-new-tokens", type=int, default=1024)
    g.add_argument("--ignore-eos", action="store_true")
    g.add_argument("--return-logprob", action="store_true", help="/generate only: return_logprob + top_logprobs_num 1")
    g.add_argument("--concurrency", type=int, default=1)
    g.add_argument("--timeout", type=float, default=7200)
    g.add_argument("--stream", action="store_true")
    g.add_argument("--keep-itl", action="store_true", help="keep the full ITL list (streaming)")
    g.add_argument("--model", default="default", help="model name for chat API")
    g.set_defaults(fn=cmd_generate)
    d = sub.add_parser("diff", help="compare two outputs.jsonl files")
    d.add_argument("--a", required=True)
    d.add_argument("--b", required=True)
    d.add_argument("--report", default="report.json")
    d.add_argument("--expect-tokens", type=int, default=None)
    d.add_argument("--max-4gram-repeats", type=int, default=50)
    d.add_argument("--exact-prefix-tokens", type=int, default=256)
    d.add_argument("--min-agreement", type=float, default=0.95)
    d.set_defaults(fn=cmd_diff)
    args = ap.parse_args()
    sys.exit(args.fn(args))


if __name__ == "__main__":
    main()
