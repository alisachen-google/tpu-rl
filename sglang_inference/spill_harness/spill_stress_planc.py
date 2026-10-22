#!/usr/bin/env python3
"""Hardening probes for the decode-HiCache reverse pull (Plan C) on a running 1P/1D pair.

Phases (each bounded by --phase-timeout; a phase that does not finish is HANG):
  concurrent     : N sessions, flush the prefill only, N recalls at once
  abort          : drop the client 0.5 s into a recall, re-issue
  prefill_flush  : flush the prefill while a recall is in flight (tree reset under a pull)
  decode_pause   : SIGSTOP the decode's TP0 scheduler during a recall for --pause-hold s
                   (the pull must time out or settle, never hang), then SIGCONT and recall again
  decode_flush   : flush both engines, recall must miss cleanly and the next recall must work
  offer_race     : send turn 2 the instant turn 1 returns, racing the decode's prefix offer
  soak           : --soak-min minutes of agentic churn with health probes
Health of both engines is polled every 5 s throughout.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spill_bench import flush, generate, load_ids  # noqa: E402
from spill_stress import HealthWatch, bounded  # noqa: E402

ME = os.environ.get("USER", "alisachen_google_com")


def decode_tp0_pid():
    out = subprocess.run(["pgrep", "-u", ME, "-f", "^[^ ]*sglang::scheduler_TP0"], capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lb", default="http://10.128.15.217:8080")
    ap.add_argument("--prefill-url", default="http://10.128.15.217:30000")
    ap.add_argument("--decode-url", default="http://10.128.0.90:30000")
    ap.add_argument("--prompts", type=Path, default=Path.home() / "pd_disagg/prompts/gpt-oss")
    ap.add_argument("--tokenizer", default="/mnt/data/tt/hf-cache/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a")
    ap.add_argument("--prefix", type=int, default=32768)
    ap.add_argument("--new", type=int, default=8192)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--phases", default="concurrent,abort_compute,prefill_flush,decode_pause,decode_flush,offer_race,abort")
    ap.add_argument("--spill-wait", type=float, default=5)
    ap.add_argument("--pause-hold", type=float, default=30)
    ap.add_argument("--soak-min", type=float, default=10)
    ap.add_argument("--phase-timeout", type=float, default=1500)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    src = args.prompts / f"{args.prefix}.jsonl"
    prefix_ids = load_ids(tok, src if src.exists() else args.prompts / "32768.jsonl", args.prefix)
    new_ids = load_ids(tok, args.prompts / "8192.jsonl", args.new, row=1)
    q = tok("\n\nContinue the story.", add_special_tokens=False)["input_ids"]
    engines = [args.prefill_url, args.decode_url]
    report = {"args": {k: str(v) for k, v in vars(args).items()}, "phases": {}}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def session_ids(i):
        salt = tok(f"Stress session {uuid.uuid4().hex} {i}\n\n", add_special_tokens=False)["input_ids"]
        t1 = salt + prefix_ids + q
        return t1, t1 + new_ids + q

    def gen(ids, timeout=None):
        return generate(args.lb, ids, args.max_new_tokens, timeout or args.phase_timeout)

    def summarize(res):
        return {k: res.get(k) for k in ("ttft_s", "e2e_s", "cached_tokens", "cached_tokens_details", "completion_tokens", "finish_reason")}

    def run_phase(name, fn):
        r, st, dt = bounded(fn, args.phase_timeout, name)
        report["phases"][name] = {"status": st, "seconds": round(dt, 1), "result": r}
        print(json.dumps({"phase": name, "status": st, "seconds": round(dt, 1)}), flush=True)

    with HealthWatch(engines, 2.0) as hw:
        phases = [p.strip() for p in args.phases.split(",") if p.strip()]

        if "concurrent" in phases:
            def phase():
                sessions = [session_ids(i) for i in range(args.sessions)]
                with ThreadPoolExecutor(args.sessions) as ex:
                    first = list(ex.map(lambda s: gen(s[0]), sessions))
                time.sleep(args.spill_wait)
                flush([args.prefill_url], 600)
                with ThreadPoolExecutor(args.sessions) as ex:
                    second = list(ex.map(lambda s: gen(s[1]), sessions))
                return {"turn1": [summarize(r) for r in first], "turn2": [summarize(r) for r in second]}
            run_phase("concurrent", phase)

        if "abort_compute" in phases:
            # Abort while the prefill is still computing a fresh prompt (no
            # prefix anywhere): the request may sit at the gate or be mid
            # chunked prefill. The engines must drop it cleanly and a flush
            # must succeed right after.
            def phase():
                fresh, _ = session_ids(77)  # a fresh prefix nothing holds yet
                body = {"input_ids": fresh, "sampling_params": {"temperature": 0, "max_new_tokens": args.max_new_tokens, "ignore_eos": True}, "stream": True}
                try:
                    with requests.post(args.lb.rstrip("/") + "/generate", json=body, stream=True, timeout=(10, 1.0)) as r:
                        for _ in r.iter_lines():
                            break
                except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
                    pass  # the client dropped the stream on purpose
                time.sleep(2)
                flush([args.prefill_url], 120)
                b = gen(fresh)
                return {"after_abort": summarize(b)}
            run_phase("abort_compute", phase)


        if "prefill_flush" in phases:
            def phase():
                t1, t2 = session_ids(88)
                a = gen(t1)
                time.sleep(args.spill_wait)
                flush([args.prefill_url], 600)
                box = {}

                def go():
                    box["b"] = gen(t2)

                th = threading.Thread(target=go, daemon=True)
                th.start()
                time.sleep(0.3)  # the pull is in flight now
                flush([args.prefill_url], 600)  # reset the tree under it
                th.join(args.phase_timeout)
                if th.is_alive():
                    raise RuntimeError("turn 2 did not return after a mid-pull prefill flush")
                c = gen(session_ids(89)[0])
                return {"turn1": summarize(a), "turn2_flushed_midpull": summarize(box["b"]), "fresh_after": summarize(c)}
            run_phase("prefill_flush", phase)

        if "decode_pause" in phases:
            def phase():
                pid = decode_tp0_pid()
                if pid is None:
                    return {"skipped": "decode scheduler not on this host"}
                t1, t2 = session_ids(77)
                a = gen(t1)
                time.sleep(args.spill_wait)
                flush([args.prefill_url], 600)
                box = {}

                def go():
                    box["b"] = gen(t2)

                th = threading.Thread(target=go, daemon=True)
                th.start()
                time.sleep(0.2)
                os.kill(pid, signal.SIGSTOP)
                t_stop = time.time()
                try:
                    time.sleep(args.pause_hold)
                finally:
                    os.kill(pid, signal.SIGCONT)
                th.join(args.phase_timeout)
                if th.is_alive():
                    raise RuntimeError("turn 2 did not return after the decode pause")
                time.sleep(args.spill_wait)
                flush([args.prefill_url], 600)
                c = gen(t2)
                return {"paused_pid": pid, "paused_s": round(time.time() - t_stop, 1), "turn2_during_pause": summarize(box["b"]), "recall_after_resume": summarize(c)}
            run_phase("decode_pause", phase)

        if "decode_flush" in phases:
            def phase():
                t1, t2 = session_ids(66)
                a = gen(t1)
                time.sleep(args.spill_wait)
                flush(engines, 600)
                b = gen(t2)
                t3, t4 = session_ids(67)
                c = gen(t3)
                time.sleep(args.spill_wait)
                flush([args.prefill_url], 600)
                d = gen(t4)
                return {"turn1": summarize(a), "turn2_after_both_flushed": summarize(b), "turn1_next": summarize(c), "turn2_recall": summarize(d)}
            run_phase("decode_flush", phase)

        if "offer_race" in phases:
            def phase():
                t1, t2 = session_ids(55)
                a = gen(t1)
                flush([args.prefill_url], 600)
                b = gen(t2)  # right away: the offer may still be landing
                return {"turn1": summarize(a), "turn2_immediate": summarize(b)}
            run_phase("offer_race", phase)

        if "abort" in phases:
            # Abort 0.5 s into a recall: with a fast pull this lands while
            # the prefill-to-decode transfer is in flight. Raiden keeps the
            # prefill's inflight entry until its own timeout, which blocks a
            # deferred flush; run this phase last.
            def phase():
                t1, t2 = session_ids(99)
                a = gen(t1)
                time.sleep(args.spill_wait)
                flush([args.prefill_url], 600)
                body = {"input_ids": t2, "sampling_params": {"temperature": 0, "max_new_tokens": args.max_new_tokens, "ignore_eos": True}, "stream": True}
                try:
                    with requests.post(args.lb.rstrip("/") + "/generate", json=body, stream=True, timeout=(10, 0.5)) as r:
                        for _ in r.iter_lines():
                            break
                except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError):
                    pass  # the client dropped the stream on purpose
                time.sleep(2)
                b = gen(t2)
                return {"turn1": summarize(a), "reissued": summarize(b)}
            run_phase("abort", phase)

        if "soak" in phases:
            def phase():
                deadline = time.time() + args.soak_min * 60
                rounds, errors = 0, 0
                while time.time() < deadline:
                    sessions = [session_ids(1000 + rounds * 10 + i) for i in range(args.sessions)]
                    with ThreadPoolExecutor(args.sessions) as ex:
                        list(ex.map(lambda s: gen(s[0]), sessions))
                    time.sleep(1.0)
                    with ThreadPoolExecutor(args.sessions) as ex:
                        res = list(ex.map(lambda s: gen(s[1]), sessions))
                    errors += sum(1 for r in res if r.get("finish_reason") is None)
                    rounds += 1
                    if rounds % 3 == 0:
                        flush([args.prefill_url], 600)
                return {"rounds": rounds, "errors": errors}
            run_phase("soak", phase)

    report["health"] = {"samples": hw.samples, "slow": hw.slow_events[:50], "errors": hw.errors[:50]}
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"health_samples": hw.samples, "slow_probes": len(hw.slow_events), "probe_errors": len(hw.errors)}), flush=True)


if __name__ == "__main__":
    main()
