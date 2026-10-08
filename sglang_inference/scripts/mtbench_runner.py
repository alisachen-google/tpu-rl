#!/usr/bin/env python3
"""mtbench_runner.py - MT-Bench (80 questions x 2 turns) through /generate with the model's chat template.
Turn 2 re-sends turn 1's prompt and answer, so it measures prefix caching (cached_tokens, TTFT) and, on a speculative server,
acceptance (completion_tokens / spec_verify_ct). Greedy, capped output.
Usage: mtbench_runner.py --url URL --questions question.jsonl --snapshot <hf snapshot dir> [--n 80] [--concurrency 1] [--max-new-tokens 512] --out out.jsonl
"""
import argparse, json, os, re, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0"); os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import requests
from transformers import AutoTokenizer

def final_text(raw: str) -> str:
    """Assistant content for the next turn: the final channel if Harmony channels are present, else the raw text."""
    if "<|channel|>final<|message|>" in raw:
        raw = raw.split("<|channel|>final<|message|>", 1)[1]
    for stop in ("<|return|>", "<|end|>", "<|call|>"):
        raw = raw.split(stop, 1)[0]
    return raw.strip()

def gen(url, text, max_new_tokens, timeout):
    t0 = time.time()
    r = requests.post(url.rstrip("/") + "/generate", json={"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens}}, timeout=timeout)
    dt = time.time() - t0
    if r.status_code != 200:
        return {"status": r.status_code, "error": r.text[:200], "e2e_s": dt}, ""
    d = r.json(); m = d.get("meta_info", {})
    rec = {"status": 200, "e2e_s": round(dt, 3), "prompt_tokens": m.get("prompt_tokens"), "cached_tokens": m.get("cached_tokens"),
           "completion_tokens": m.get("completion_tokens"), "spec_verify_ct": m.get("spec_verify_ct"), "ttft_s": m.get("first_token_latency"),
           "finish_reason": (m.get("finish_reason") or {}).get("type") if isinstance(m.get("finish_reason"), dict) else m.get("finish_reason")}
    return rec, d.get("text", "")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--questions", required=True); ap.add_argument("--snapshot", required=True)
    ap.add_argument("--n", type=int, default=80); ap.add_argument("--concurrency", type=int, default=1); ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--timeout", type=float, default=900); ap.add_argument("--out", required=True); ap.add_argument("--label", default="")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.snapshot)
    qs = [json.loads(l) for l in open(a.questions)][: a.n]
    def run_q(q):
        u1, u2 = q["turns"][0], q["turns"][1]
        p1 = tok.apply_chat_template([{"role": "user", "content": u1}], tokenize=False, add_generation_prompt=True)
        r1, t1 = gen(a.url, p1, a.max_new_tokens, a.timeout)
        a1 = final_text(t1) or "(no answer)"
        p2 = tok.apply_chat_template([{"role": "user", "content": u1}, {"role": "assistant", "content": a1}, {"role": "user", "content": u2}], tokenize=False, add_generation_prompt=True)
        r2, t2 = gen(a.url, p2, a.max_new_tokens, a.timeout)
        return {"question_id": q["question_id"], "category": q["category"], "turn1": r1, "turn2": r2, "shared_prefix_chars": len(os.path.commonprefix([p1, p2]))}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        rows = list(ex.map(run_q, qs))
    wall = time.time() - t0
    with open(a.out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    def col(turn, k): return [r[turn][k] for r in rows if r[turn].get("status") == 200 and r[turn].get(k) is not None]
    def mean(x): return statistics.mean(x) if x else float("nan")
    def p50(x): return statistics.median(x) if x else float("nan")
    ok1 = sum(r["turn1"].get("status") == 200 for r in rows); ok2 = sum(r["turn2"].get("status") == 200 for r in rows)
    out_tokens = sum(col("turn1", "completion_tokens")) + sum(col("turn2", "completion_tokens"))
    cached2 = col("turn2", "cached_tokens"); prompt2 = col("turn2", "prompt_tokens")
    cache_frac = (sum(cached2) / sum(prompt2)) if prompt2 and sum(prompt2) else float("nan")
    acc = [r[t]["completion_tokens"] / r[t]["spec_verify_ct"] for r in rows for t in ("turn1", "turn2") if r[t].get("status") == 200 and r[t].get("spec_verify_ct")]
    print(f"[{a.label}] c={a.concurrency} questions={len(rows)} ok turn1={ok1} turn2={ok2} wall={wall:.1f}s output_tokens={out_tokens} out_tok/s={out_tokens/wall:.1f}")
    print(f"[{a.label}]   TTFT turn1 mean={mean(col('turn1','ttft_s'))*1000:.0f}ms p50={p50(col('turn1','ttft_s'))*1000:.0f}ms | turn2 mean={mean(col('turn2','ttft_s'))*1000:.0f}ms p50={p50(col('turn2','ttft_s'))*1000:.0f}ms")
    print(f"[{a.label}]   turn2 cached_tokens mean={mean(cached2):.0f} of prompt mean={mean(prompt2):.0f} (cached fraction {cache_frac:.2f}); turn1 cached mean={mean(col('turn1','cached_tokens')):.0f}")
    print(f"[{a.label}]   e2e per turn mean: turn1={mean(col('turn1','e2e_s')):.2f}s turn2={mean(col('turn2','e2e_s')):.2f}s; completion mean turn1={mean(col('turn1','completion_tokens')):.0f} turn2={mean(col('turn2','completion_tokens')):.0f}")
    if acc: print(f"[{a.label}]   spec accept_length mean={mean(acc):.3f} p50={p50(acc):.3f} min={min(acc):.3f} max={max(acc):.3f} (n={len(acc)})")
    else: print(f"[{a.label}]   no spec_verify_ct (non-speculative server)")

if __name__ == "__main__":
    main()
