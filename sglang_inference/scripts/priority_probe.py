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

def hold_slots(url, n, prio, tokens, results, stagger=0):
    """Occupy n slots with long decodes of the given priority (streaming so TTFT is observable).
    With stagger > 0 the holders have lengths tokens, tokens+stagger, ... so slots free one at a time."""
    ths = []
    for i in range(n):
        def w(i=i):
            results[f"hold{i}"] = gen(url, f"Write a very long story about slot {i}. " * 20, priority=prio, max_new_tokens=tokens + stagger * i, stream=True, timeout=1800)
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
        r = gen(U, "Hello.", priority=5); say("A5-abort", r["status"] in (400, 503), f"priority with scheduling disabled + abort flag -> HTTP {r['status']} {str(r.get('body'))[:100]}")
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
            # C1: diff == threshold must NOT preempt (strict >): the newcomer waits for a slot.
            r = gen(U, "Urgent question: what is 2+2?", priority=a.threshold, max_new_tokens=8, stream=True, timeout=900)
            r1 = retract_lines(D); say("C1-threshold", (r1 or 0) == (r0 or 0) and (r["ttft"] or 0) > 2, f"priority {a.threshold} vs running 0 (diff = threshold): retracted {r0}->{r1}, ttft={r['ttft'] and round(r['ttft'],1)}s (waited for a slot)")
            # C2 needs every slot busy again: wait for the first holder set, start a fresh one, then send diff = threshold+1.
            for t in ths: t.join(timeout=1500)
            res2 = {}; ths = hold_slots(U, a.slots, 0, a.hold_tokens, res2); time.sleep(12)
            running = metric(D, "num_running_reqs", 'tp_rank="0"'); r1 = retract_lines(D)
            r = gen(U, "Urgent question: what is 3+3?", priority=a.threshold + 1, max_new_tokens=8, stream=True, timeout=900)
            r2 = retract_lines(D)
            say("C2-preempt", (r["ttft"] or 1e9) < 3 and r["status"] == 200, f"priority {a.threshold+1} (diff = threshold+1) at running={running}: ttft={r['ttft'] and round(r['ttft'],1)}s (immediate = a running decode was preempted), retracted gauge {r1}->{r2}, HTTP {r['status']}")
            for t in ths: t.join(timeout=1800)
            bad = [k for k, v in res2.items() if v.get("status") != 200 or (v.get("completion_tokens") not in (None, a.hold_tokens))]
            say("C2-victims-resume", not [k for k, v in res2.items() if v.get("status") != 200], f"second holder set finished: {sum(1 for v in res2.values() if v.get('status')==200)}/{len(res2)} ok (preempted request resumed and completed)")
        else:
            r = gen(U, "Urgent question: what is 2+2?", priority=100, max_new_tokens=8, stream=True, timeout=900)
            r1 = retract_lines(D); say("B1-no-preempt", (r1 or 0) == (r0 or 0) and r["status"] == 200, f"priority 100 arrived at full occupancy: retracted {r0}->{r1}, ttft={r['ttft'] and round(r['ttft'],1)}s (waited for a free slot), HTTP {r['status']}")
        for t in ths: t.join(timeout=1500)
        bad = [k for k, v in res.items() if v.get("status") != 200]; say("B1-holders", not bad, f"held requests finished: {len(res)-len(bad)}/{len(res)} ok {bad[:3]}")
        # B2/B3/B4: ordering of waiting requests. Slots must free one at a time (staggered holder lengths), otherwise every
        # waiting request is admitted in the same batch and TTFT order says nothing about the queue order.
        # Holders: 2500, 3000, ... tokens so the first slot frees well after every probe is queued and later slots free
        # several seconds apart; admission time = sent + ttft (the first token is emitted when the decode admits the request).
        res = {}; t_hold = time.time(); ths = hold_slots(U, a.slots, 0, 2500, res, stagger=500); time.sleep(12)
        order = {}; lock = threading.Lock(); ths2 = []
        prios = [0, 1, 2, 3, 4, 5, 6, 7]
        def adm(r): return (r["sent"] + r["ttft"] - t_hold) if r.get("status") == 200 and r.get("ttft") is not None else None
        for p in prios:
            def w(p=p):
                r = gen(U, f"Ordering probe priority {p}. Count to three.", priority=p, max_new_tokens=8, stream=True, timeout=1800)
                with lock: order[p] = adm(r)
            t = threading.Thread(target=w, daemon=True); t.start(); ths2.append(t); time.sleep(0.2)
        same = {}
        for k in range(4):
            def w2(k=k):
                r = gen(U, f"Tie probe {k}. Count to three.", priority=3, max_new_tokens=8, stream=True, timeout=1800)
                with lock: same[k] = adm(r)
            t = threading.Thread(target=w2, daemon=True); t.start(); ths2.append(t); time.sleep(0.2)
        q_done = time.time() - t_hold
        for t in ths2: t.join(timeout=1800)
        adm_all = sorted([(t, f"p{p}") for p, t in order.items() if t is not None] + [(t, f"tie{k}") for k, t in same.items() if t is not None])
        say("B2-admissions", None, f"all probes queued by t={q_done:.1f}s; admissions (s after holders started): " + " ".join(f"{n}@{t:.1f}" for t, n in adm_all))
        got = [p for p, _ in sorted(((p, t) for p, t in order.items() if t is not None), key=lambda x: x[1])]
        expect = sorted(prios, reverse=(a.mode != "low"))
        say("B2-order" if a.mode != "low" else "B4-order-low", got == expect, f"admission order of distinct priorities: {got} (expected {expect})")
        tie = [k for k, _ in sorted(((k, t) for k, t in same.items() if t is not None), key=lambda x: x[1])]
        say("B3-fifo-tie", tie == sorted(tie), f"equal-priority admission order: {tie} (FIFO expected)")
        for t in ths: t.join(timeout=1800)
        bad = [k for k, v in res.items() if v.get("status") != 200]; say("B2-holders", not bad, f"staggered holders finished: {len(res)-len(bad)}/{len(res)} ok")

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
