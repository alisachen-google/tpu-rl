#!/usr/bin/env python3
"""Two-turn prefix-reuse benchmark for the decode-host DRAM spill/recall feature (1P/1D).

Turn 1: [salt] + prefix(P tokens) + question  -> 1k output   (cold prefill; spills to the decode-host store)
Turn 2: turn-1 prompt + new(N tokens) + question -> 1k output
Scenario decides what happens between the turns:
  local     : nothing                      -> turn 2 hits prefill HBM/host DRAM (upper bound)
  flush     : /flush_cache on both engines -> turn 2 recomputes (no storage) or recalls from the
              decode-host DRAM store (storage enabled). The label is given by --label.
Token-exact prompts are sent as input_ids. Every repetition uses a fresh salt so turn 1 is always cold.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import uuid
from pathlib import Path

import requests

METRIC_RE = re.compile(r'^(sglang:[a-z_]+)(\{[^}]*\})?\s+([0-9.e+-]+)$', re.M)


def load_ids(tok, path: Path, n: int, row: int = 0):
    rows = [json.loads(l) for l in path.open()]
    text = rows[row % len(rows)]["text"]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    while len(ids) < n:  # pad with another row if the file is short
        row += 1
        ids += tok(rows[row % len(rows)]["text"], add_special_tokens=False)["input_ids"]
    return ids[:n]


def scrape(url: str) -> dict:
    try:
        r = requests.get(url.rstrip("/") + "/metrics", timeout=10)
        r.raise_for_status()
    except Exception:
        return {}
    out = {}
    for name, labels, value in METRIC_RE.findall(r.text):
        if name in ("sglang:prompt_tokens_total", "sglang:cached_tokens_total", "sglang:num_requests_total"):
            out[name] = out.get(name, 0.0) + float(value)
    return out


def flush(urls, timeout):
    for u in urls:
        r = requests.post(u.rstrip("/") + "/flush_cache", params={"timeout": 60}, timeout=timeout)
        r.raise_for_status()


def generate(lb: str, input_ids, max_new_tokens: int, timeout: float) -> dict:
    body = {
        "input_ids": input_ids,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens, "ignore_eos": True},
        "stream": True,
    }
    t0 = time.perf_counter()
    ttft = None
    last = None
    n_chunks = 0
    with requests.post(lb.rstrip("/") + "/generate", json=body, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[5:].strip()
            if payload == b"[DONE]":
                break
            chunk = json.loads(payload)
            n_chunks += 1
            if ttft is None:
                ttft = time.perf_counter() - t0
            last = chunk
    e2e = time.perf_counter() - t0
    meta = (last or {}).get("meta_info", {})
    comp = meta.get("completion_tokens")
    return {
        "ttft_s": ttft,
        "e2e_s": e2e,
        "decode_tok_s": (comp - 1) / (e2e - ttft) if comp and ttft and e2e > ttft else None,
        "prompt_tokens": meta.get("prompt_tokens"),
        "completion_tokens": comp,
        "cached_tokens": meta.get("cached_tokens"),
        "cached_tokens_details": meta.get("cached_tokens_details"),
        "finish_reason": meta.get("finish_reason"),
        "chunks": n_chunks,
        "text_head": (last or {}).get("text", "")[:80],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lb", default="http://10.128.15.217:8080")
    ap.add_argument("--prefill-url", default="http://10.128.15.217:30000")
    ap.add_argument("--decode-url", default="http://10.128.0.90:30000")
    ap.add_argument("--prompts", type=Path, default=Path.home() / "pd_disagg/prompts/gpt-oss")
    ap.add_argument("--tokenizer", default="/mnt/data/tt/hf-cache/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a")
    ap.add_argument("--prefix", type=int, required=True, help="prefix tokens, e.g. 131072 or 262144")
    ap.add_argument("--new", type=int, required=True, help="new prefill tokens in turn 2, e.g. 8192 or 16384")
    ap.add_argument("--scenario", choices=["local", "flush"], required=True)
    ap.add_argument("--flush-targets", choices=["both", "prefill"], default="both",
                    help="flush both engines (Plan B) or only the prefill (Plan C keeps the decode cache)")
    ap.add_argument("--label", required=True, help="recompute | recall | local (how to label this run)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--spill-wait", type=float, default=0.0, help="seconds to wait after turn 1 before flushing")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--run", required=True, help="server run name (for the record)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    prefix_file = args.prompts / f"{args.prefix}.jsonl"
    new_file = args.prompts / ("32768.jsonl" if args.new > 8192 else "8192.jsonl")
    prefix_ids = load_ids(tok, prefix_file, args.prefix)
    new_ids = load_ids(tok, new_file, args.new, row=1)
    q1 = tok("\n\nSummarize the key events above in one paragraph.", add_special_tokens=False)["input_ids"]
    q2 = tok("\n\nNow, taking all of the text above into account, continue the story in detail.", add_special_tokens=False)["input_ids"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for rep in range(args.repeat):
        salt = tok(f"Session {uuid.uuid4().hex} {random.random():.6f}\n\n", add_special_tokens=False)["input_ids"]
        turn1 = salt + prefix_ids + q1
        turn2 = turn1 + new_ids + q2
        rec = {
            "run": args.run, "label": args.label, "scenario": args.scenario, "rep": rep,
            "prefix": args.prefix, "new": args.new, "turn1_tokens": len(turn1), "turn2_tokens": len(turn2),
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        m0 = scrape(args.prefill_url)
        rec["turn1"] = generate(args.lb, turn1, args.max_new_tokens, args.timeout)
        m1 = scrape(args.prefill_url)
        rec["turn1"]["prefill_metrics_delta"] = {k: m1.get(k, 0) - m0.get(k, 0) for k in m1}
        print(json.dumps({"rep": rep, "turn1": {k: rec["turn1"][k] for k in ("ttft_s", "e2e_s", "cached_tokens", "cached_tokens_details", "completion_tokens")}}), flush=True)
        if args.spill_wait > 0:
            time.sleep(args.spill_wait)
        if args.scenario == "flush":
            t = time.perf_counter()
            flush([args.prefill_url] if args.flush_targets == 'prefill' else [args.prefill_url, args.decode_url], args.timeout)
            rec["flush_s"] = time.perf_counter() - t
        m2 = scrape(args.prefill_url)
        rec["turn2"] = generate(args.lb, turn2, args.max_new_tokens, args.timeout)
        m3 = scrape(args.prefill_url)
        rec["turn2"]["prefill_metrics_delta"] = {k: m3.get(k, 0) - m2.get(k, 0) for k in m3}
        print(json.dumps({"rep": rep, "turn2": {k: rec["turn2"][k] for k in ("ttft_s", "e2e_s", "cached_tokens", "cached_tokens_details", "completion_tokens", "prefill_metrics_delta")}}), flush=True)
        with args.out.open("a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
