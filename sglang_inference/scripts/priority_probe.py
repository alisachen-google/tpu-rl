#!/usr/bin/env python3
"""priority_probe.py - boundary cases for --enable-priority-scheduling on a PD pair (issue #1161).
Each case prints one line: [case] PASS|FAIL|INFO details. Metrics are read from the decode engine.
  --mode np   : no-preemption configuration (A1-A5, B1-B3, B5)
  --mode pre  : preemption on, threshold T (C1, C2, C4)
  --mode q    : max_queued_requests=4 (D1-D3)
  --mode low  : low-values-first (B4)
  --mode dis  : scheduling disabled + abort_on_priority_when_disabled (A5)
"""
import argparse, json, sys, threading, time
import requests

def gen(url, text, priority=None, max_new_tokens=16, stream=False, timeout=600, extra=None):
    body = {"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens, "ignore_eos": True}}
    if priority is not None: body["priority"] = priority
    if extra: body.update(extra)
    t0 = time.time()
    try:
        r = requests.post(url.rstrip("/") + "/generate", json=body, stream=stream, timeout=timeout)
    except Exception as e:
        return {"status": "ERR", "error": str(e)[:120], "e2e": time.time() - t0}
    out = {"status": r.status_code, "sent": t0, "ttft": None, "e2e": None, "body": None}
    if r.status_code != 200:
        out["body"] = r.text[:200]; out["e2e"] = time.time() - t0; return out
    if stream:
        for line in r.iter_lines():
            if line and out["ttft"] is None: out["ttft"] = time.time() - t0
        out["e2e"] = time.time() - t0
    else:
        d = r.json(); out["ttft"] = (d.get("meta_info") or {}).get("first_token_latency"); out["e2e"] = time.time() - t0
        out["completion_tokens"] = (d.get("meta_info") or {}).get("completion_tokens"); out["text"] = d.get("text", "")[:80]
    return out

def metric(dec, name, must=None):
    try:
        txt = requests.get(dec.rstrip("/") + "/metrics", timeout=10).text
    except Exception:
        return None
    vals = [l for l in txt.splitlines() if l.startswith(f"sglang:{name}{{") and (must is None or must in l)]
    if not vals: return None
    return float(vals[0].rsplit(" ", 1)[1])

def retract_lines(dec):
    return metric(dec, "num_retracted_reqs", 'tp_rank="0"')

def hold_slots(url, n, prio, tokens, results):
    """Occupy n slots with long decodes of the given priority (streaming so TTFT is observable)."""
    ths = []
    for i in range(n):
        def w(i=i):
            results[f"hold{i}"] = gen(url, f"Write a very long story about slot {i}. " * 20, priority=prio, max_new_tokens=tokens, stream=True, timeout=900)
        t = threading.Thread(target=w, daemon=True); t.start(); ths.append(t)
    return ths

def say(case, ok, msg):
    print(f"[{case}] {'PASS' if ok is True else 'FAIL' if ok is False else 'INFO'} {msg}", flush=True)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--decode", required=True)
    ap.add_argument("--mode", required=True, choices=["np", "pre", "q", "low", "dis"]); ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--threshold", type=int, default=10); ap.add_argument("--hold-tokens", type=int, default=3000)
    a = ap.parse_args(); U, D = a.url, a.decode
    base = retract_lines(D)

    if a.mode == "dis":
        r = gen(U, "Hello.", priority=5); say("A5-abort", r["status"] == 400, f"priority with scheduling disabled + abort flag -> HTTP {r['status']} {str(r.get('body'))[:100]}")
        r = gen(U, "Hello."); say("A5-plain", r["status"] == 200, f"no priority -> HTTP {r['status']}")
        return

    # A. validation boundaries
    for case, prio in (("A1-absent", None), ("A3-zero", 0), ("A3-neg", -1), ("A3-max32", 2**31 - 1), ("A3-min32", -(2**31))):
        r = gen(U, "Say hi.", priority=prio); say(case, r["status"] == 200, f"priority={prio} -> HTTP {r['status']} {str(r.get('body') or '')[:80]}")
    for case, prio in (("A4-huge", 2**63), ("A4-float", 1.5), ("A4-str", "high")):
        r = gen(U, "Say hi.", priority=prio); say(case, r["status"] in (400, 422), f"priority={prio!r} -> HTTP {r['status']} {str(r.get('body') or '')[:100]}")
    hc = requests.get(D.rstrip("/") + "/health", timeout=10).status_code; say("A4-health", hc == 200, f"decode /health {hc} after malformed priorities")

    if a.mode in ("np", "low", "pre"):
        # B1 / C1 / C2: occupy all slots with low-priority long decodes, then submit a higher-priority request.
        res = {}; ths = hold_slots(U, a.slots, 0, a.hold_tokens, res); time.sleep(12)
        running = metric(D, "num_running_reqs", 'tp_rank="0"'); say("B1-setup", running is not None and running >= a.slots - 1, f"running={running} of {a.slots} slots held at priority 0")
        r0 = retract_lines(D)
        if a.mode == "pre":
            # C1: diff == threshold must NOT preempt; C2: diff == threshold+1 must preempt.
            r = gen(U, "Urgent question: what is 2+2?", priority=a.threshold, max_new_tokens=8, stream=True, timeout=900)
            r1 = retract_lines(D); say("C1-threshold", (r1 or 0) == (r0 or 0), f"priority {a.threshold} vs running 0 (diff = threshold): retracted {r0}->{r1}, ttft={r['ttft'] and round(r['ttft'],1)}s")
            r = gen(U, "Urgent question: what is 3+3?", priority=a.threshold + 1, max_new_tokens=8, stream=True, timeout=900)
            r2 = retract_lines(D); say("C2-preempt", (r2 or 0) > (r1 or 0) or (r["ttft"] or 1e9) < 15, f"priority {a.threshold+1} (diff = threshold+1): retracted {r1}->{r2}, ttft={r['ttft'] and round(r['ttft'],1)}s, HTTP {r['status']}")
        else:
            r = gen(U, "Urgent question: what is 2+2?", priority=100, max_new_tokens=8, stream=True, timeout=900)
            r1 = retract_lines(D); say("B1-no-preempt", (r1 or 0) == (r0 or 0) and r["status"] == 200, f"priority 100 arrived at full occupancy: retracted {r0}->{r1}, ttft={r['ttft'] and round(r['ttft'],1)}s (waited for a free slot), HTTP {r['status']}")
        # B2/B3/B4: ordering of waiting requests while slots are busy.
        order = {}; lock = threading.Lock(); ths2 = []
        prios = [0, 1, 2, 3, 4, 5, 6, 7] if a.mode != "np" or True else []
        for p in prios:
            def w(p=p):
                r = gen(U, f"Ordering probe priority {p}. Count to three.", priority=p, max_new_tokens=8, stream=True, timeout=900)
                with lock: order[p] = r["ttft"] if r["status"] == 200 else None
            t = threading.Thread(target=w, daemon=True); t.start(); ths2.append(t); time.sleep(0.3)
        same = {}
        for k in range(4):
            def w2(k=k):
                r = gen(U, f"Tie probe {k}. Count to three.", priority=3, max_new_tokens=8, stream=True, timeout=900)
                with lock: same[k] = r["ttft"] if r["status"] == 200 else None
            t = threading.Thread(target=w2, daemon=True); t.start(); ths2.append(t); time.sleep(0.3)
        for t in ths2: t.join(timeout=1200)
        got = [p for p, _ in sorted(((p, t) for p, t in order.items() if t is not None), key=lambda x: x[1])]
        expect = sorted(prios, reverse=(a.mode != "low")); first = got[: len(expect)]
        say("B2-order" if a.mode != "low" else "B4-order-low", first == expect, f"admission order by TTFT: {got} (expected {expect})")
        tie = [k for k, _ in sorted(((k, t) for k, t in same.items() if t is not None), key=lambda x: x[1])]
        say("B3-fifo-tie", tie == sorted(tie), f"equal priority admission order: {tie} (FIFO expected)")
        for t in ths: t.join(timeout=1500)
        bad = [k for k, v in res.items() if v.get("status") != 200]; say("B1-holders", not bad, f"held requests finished: {len(res)-len(bad)}/{len(res)} ok {bad[:3]}")

    if a.mode == "q":
        res = {}; ths = hold_slots(U, a.slots, 5, a.hold_tokens, res); time.sleep(12)
        waiting = {}; lock = threading.Lock(); ths2 = []
        for k in range(4):  # fill the waiting queue (max_queued_requests=4) with priority 5
            def w(k=k):
                r = gen(U, f"Queue filler {k}.", priority=5, max_new_tokens=8, timeout=900)
                with lock: waiting[k] = r
            t = threading.Thread(target=w, daemon=True); t.start(); ths2.append(t); time.sleep(0.2)
        time.sleep(3)
        r = gen(U, "Low newcomer.", priority=3, max_new_tokens=8, timeout=120); say("D1-low-rejected", r["status"] == 503, f"newcomer priority 3 into a full queue of 5s -> HTTP {r['status']} {str(r.get('body') or '')[:80]}")
        r = gen(U, "Equal newcomer.", priority=5, max_new_tokens=8, timeout=120); say("D3-equal-rejected", r["status"] == 503, f"newcomer priority 5 (equal) -> HTTP {r['status']}")
        def hi():
            waiting["hi"] = gen(U, "High newcomer.", priority=9, max_new_tokens=8, timeout=900)
        th = threading.Thread(target=hi, daemon=True); th.start(); time.sleep(5)
        evicted = [k for k, v in waiting.items() if k != "hi" and v.get("status") == 503]
        for t in ths2: t.join(timeout=1200)
        th.join(timeout=1200)
        evicted = [k for k, v in waiting.items() if k != "hi" and v.get("status") == 503]
        say("D2-evict-lowest", len(evicted) == 1 and waiting.get("hi", {}).get("status") == 200, f"newcomer priority 9: evicted {evicted} (expect exactly one, the newest priority-5 filler = 3), newcomer HTTP {waiting.get('hi',{}).get('status')} msg={str([v.get('body') for k,v in waiting.items() if k in evicted])[:120]}")
        for t in ths: t.join(timeout=1500)
    final = retract_lines(D); say("end-retractions", None, f"num_retracted_reqs gauge {base} -> {final}")
    used = metric(D, "num_used_tokens", 'tp_rank="0"'); say("end-idle", used is not None and used < 1024, f"decode num_used_tokens at end = {used}")

if __name__ == "__main__":
    main()
