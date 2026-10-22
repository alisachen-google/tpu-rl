#!/usr/bin/env python3
"""Agentic multi-session workload for the decode-host DRAM spill feature (1P/1D).

N sessions run at the same time. Each session has its own long context (prefix tokens, salted so
sessions never share pages). Each turn appends a tool-result block (new tokens) and a short
question, then generates a short answer. Sessions pause a random think time between turns, so
they interleave and churn the prefill caches like real multi-user agent traffic.

Per turn we record TTFT, end-to-end time and where the cached prefix came from
(device / host / storage). The summary gives p50/p95 TTFT per turn index and the share of
prefix tokens served by each tier.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import threading
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spill_bench import generate, load_ids  # noqa: E402


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((len(s) - 1) * p))))
    return s[k]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lb", default="http://10.128.15.217:8080")
    ap.add_argument("--prompts", type=Path, default=Path.home() / "pd_disagg/prompts/gpt-oss")
    ap.add_argument("--tokenizer", default="/mnt/data/tt/hf-cache/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a")
    ap.add_argument("--sessions", type=int, default=8)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--prefix", type=int, default=131072, help="private context tokens per session")
    ap.add_argument("--shared", type=int, default=0, help="shared system/tool prefix tokens, identical for every session")
    ap.add_argument("--turn-new", type=int, default=4096)
    ap.add_argument("--turn-out", type=int, default=512)
    ap.add_argument("--think-min", type=float, default=2.0)
    ap.add_argument("--think-max", type=float, default=8.0)
    ap.add_argument("--stagger", type=float, default=3.0, help="seconds between session starts")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--run", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    sizes = sorted(int(f.stem) for f in args.prompts.glob("*.jsonl") if f.stem.isdigit())
    pick = next((n for n in sizes if n >= args.prefix), sizes[-1])  # smallest token-exact file that covers the prefix
    prefix_src = args.prompts / f"{pick}.jsonl"
    prefix_ids = load_ids(tok, prefix_src, args.prefix)
    shared_ids = load_ids(tok, args.prompts / "32768.jsonl", args.shared, row=0) if args.shared else []
    shared_ids = tok("System: you are a coding agent with tools read_file, run_tests, grep.\n", add_special_tokens=False)["input_ids"] + shared_ids
    tool_src = args.prompts / "32768.jsonl"
    tool_rows = [json.loads(l) for l in tool_src.open()]
    tool_pool = tok(tool_rows[1 % len(tool_rows)]["text"], add_special_tokens=False)["input_ids"]
    q_ids = tok("\n\nBased on everything above, what should the agent do next? Answer briefly.", add_special_tokens=False)["input_ids"]
    rng = random.Random(args.seed)
    records = []
    lock = threading.Lock()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def session(sid):
        srng = random.Random(args.seed * 100 + sid)
        salt = tok(f"Agent session {uuid.uuid4().hex} user {sid}\n\n", add_special_tokens=False)["input_ids"]
        ctx = shared_ids + salt + prefix_ids  # shared part first so every user can hit it
        t_start = time.perf_counter()
        for turn in range(args.turns):
            if turn > 0:
                off = srng.randrange(0, max(1, len(tool_pool) - args.turn_new))
                tool_block = tok(f"\n\n[tool result {turn}]\n", add_special_tokens=False)["input_ids"] + tool_pool[off : off + args.turn_new]
                ctx = ctx + tool_block
            prompt = ctx + q_ids
            t0 = time.perf_counter()
            try:
                res = generate(args.lb, prompt, args.turn_out, args.timeout)
                err = None
            except Exception as exc:  # noqa: BLE001
                res, err = {}, repr(exc)[:200]
            rec = {
                "run": args.run, "label": args.label, "session": sid, "turn": turn,
                "prompt_tokens": len(prompt), "queued_at_s": round(t0 - t_start, 2), "error": err,
                **{k: res.get(k) for k in ("ttft_s", "e2e_s", "cached_tokens", "cached_tokens_details", "completion_tokens", "decode_tok_s")},
            }
            with lock:
                records.append(rec)
                with args.out.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
            print(json.dumps({k: rec[k] for k in ("session", "turn", "prompt_tokens", "ttft_s", "cached_tokens", "cached_tokens_details", "error")}), flush=True)
            if err:
                break
            # the answer is produced by the decode engine; the next turn carries it as part of the context
            ctx = ctx + tok(res.get("text_head", "") or "ok", add_special_tokens=False)["input_ids"][:64]
            time.sleep(srng.uniform(args.think_min, args.think_max))
        with lock:
            records.append({"run": args.run, "label": args.label, "session": sid, "turn": -1, "session_wall_s": round(time.perf_counter() - t_start, 1)})

    threads = []
    wall0 = time.perf_counter()
    for sid in range(args.sessions):
        t = threading.Thread(target=session, args=(sid,), daemon=True)
        t.start()
        threads.append(t)
        time.sleep(args.stagger)
    for t in threads:
        t.join()
    wall = time.perf_counter() - wall0

    turns = [r for r in records if r.get("turn", -1) >= 0 and not r.get("error")]
    errors = [r for r in records if r.get("error")]
    per_turn = {}
    for r in turns:
        per_turn.setdefault(r["turn"], []).append(r["ttft_s"])
    tiers = {"device": 0, "host": 0, "storage": 0}
    prompt_total = 0
    for r in turns:
        d = r.get("cached_tokens_details") or {}
        for k in tiers:
            tiers[k] += d.get(k) or 0
        prompt_total += r["prompt_tokens"]
    later = [r["ttft_s"] for r in turns if r["turn"] > 0]
    summary = {
        "run": args.run, "label": args.label, "sessions": args.sessions, "turns": args.turns,
        "prefix": args.prefix, "shared": args.shared, "turn_new": args.turn_new, "turn_out": args.turn_out,
        "wall_s": round(wall, 1), "errors": len(errors),
        "ttft_turn0_p50": pct([r["ttft_s"] for r in turns if r["turn"] == 0], 0.5),
        "ttft_later_p50": pct(later, 0.5), "ttft_later_p95": pct(later, 0.95), "ttft_later_max": max(later) if later else None,
        "ttft_per_turn_p50": {k: round(pct(v, 0.5), 2) for k, v in sorted(per_turn.items())},
        "cached_tokens_by_tier": tiers, "prompt_tokens_total": prompt_total,
        "session_wall_p50": pct([r["session_wall_s"] for r in records if r.get("turn") == -1], 0.5),
        "output_tokens_total": sum(r.get("completion_tokens") or 0 for r in turns),
        "aggregate_output_tok_s": round(sum(r.get("completion_tokens") or 0 for r in turns) / wall, 1),
    }
    with args.out.with_suffix(".summary.json").open("a") as f:
        f.write(json.dumps(summary) + "\n")
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
