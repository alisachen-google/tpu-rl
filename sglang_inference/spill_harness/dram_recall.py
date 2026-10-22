#!/usr/bin/env python3
"""Direct test of a recall from decode DRAM: run session A, push it out of decode HBM
with N cold sessions (but keep it in the decode host pool), flush the prefill, then
send A's turn 2. Reports TTFT, cached tokens and whether the pull committed."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spill_bench import flush, generate, load_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lb", default="http://10.128.15.217:8080")
    ap.add_argument("--prefill-url", default="http://10.128.15.217:30000")
    ap.add_argument("--prompts", type=Path, default=Path.home() / "pd_disagg/prompts/gpt-oss")
    ap.add_argument("--tokenizer", default="/mnt/data/tt/hf-cache/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a")
    ap.add_argument("--prefix", type=int, default=131072)
    ap.add_argument("--new", type=int, default=8192)
    ap.add_argument("--evictors", type=int, default=7, help="cold 128k sessions to push A out of decode HBM (786k tokens)")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    prefix_ids = load_ids(tok, args.prompts / f"{args.prefix}.jsonl", args.prefix)
    new_ids = load_ids(tok, args.prompts / "8192.jsonl", args.new, row=1)
    q = tok("\n\nContinue the story.", add_special_tokens=False)["input_ids"]

    def ids(tag):
        salt = tok(f"DRAM recall {uuid.uuid4().hex} {tag}\n\n", add_special_tokens=False)["input_ids"]
        return salt + prefix_ids + q

    a1 = ids("A")
    a2 = a1 + new_ids + q
    t_a1 = generate(args.lb, a1, 128, 600)
    time.sleep(5)
    with ThreadPoolExecutor(4) as ex:
        list(ex.map(lambda i: generate(args.lb, ids(f"E{i}"), 64, 600), range(args.evictors)))
    time.sleep(10)  # let the decode write through and evict
    flush([args.prefill_url], 600)
    t0 = time.perf_counter()
    t_a2 = generate(args.lb, a2, 128, 600)
    stats = subprocess.run(["bash", str(Path.home() / "pd_disagg/spill/peer_stats.sh"), args.run], capture_output=True, text=True).stdout.strip()
    rec = {
        "run": args.run, "prefix": args.prefix, "evictors": args.evictors,
        "turn1_ttft_s": t_a1.get("ttft_s"), "turn2_ttft_s": t_a2.get("ttft_s"),
        "turn2_cached": t_a2.get("cached_tokens_details"), "turn2_wall_s": round(time.perf_counter() - t0, 2),
        "peer_stats_after": stats,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print("DRAM_RECALL " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
