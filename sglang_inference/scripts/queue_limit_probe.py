#!/usr/bin/env python3
"""queue_limit_probe.py - boundary cases for --max-queued-requests + priority eviction on the decode of a PD pair (PR #1191).
One line per case: [case] PASS|FAIL|INFO details. Metrics from the decode engine (--decode).
  --mode base    : limit 4, priority on, high values first (Q1-Q10)
  --mode low     : limit 4, low values first (L1-L2)
  --mode q0      : limit 0 (Z1-Z3)
  --mode q1      : limit 1 (O1-O3)
  --mode nops    : limit 4, priority scheduling off (N1-N2)
  --mode nolimit : no limit (U1-U2, the default path)
  --mode retract : small decode pool, long holders (R1-R3)
"""
import argparse, json, threading, time
import requests

def gen(url, text, priority=None, max_new_tokens=16, stream=False, timeout=900):
    body = {"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens, "ignore_eos": True}}
    if priority is not None: body["priority"] = priority
    t0 = time.time()
    try:
        r = requests.post(url.rstrip("/") + "/generate", json=body, stream=stream, timeout=timeout)
    except Exception as e:
        return {"status": "ERR", "error": str(e)[:160], "e2e": time.time() - t0}
    out = {"status": r.status_code, "e2e": None, "body": None, "stream_error": None}
    if r.status_code != 200:
        out["body"] = r.text[:200]; out["e2e"] = time.time() - t0; return out
    if stream:
        for line in r.iter_lines():
            if line and b"error" in line and out["stream_error"] is None: out["stream_error"] = line[:200].decode(errors="replace")
        out["e2e"] = time.time() - t0
    else:
        d = r.json(); out["e2e"] = time.time() - t0; out["completion_tokens"] = (d.get("meta_info") or {}).get("completion_tokens")
    return out

def metric(dec, name, must='tp_rank="0"'):
    try: txt = requests.get(dec.rstrip("/") + "/metrics", timeout=10).text
    except Exception: return None
    vals = [l for l in txt.splitlines() if l.startswith(f"sglang:{name}{{") and (must is None or must in l)]
    return float(vals[0].rsplit(" ", 1)[1]) if vals else None

def say(case, ok, msg): print(f"[{case}] {'PASS' if ok is True else 'FAIL' if ok is False else 'INFO'} {msg}", flush=True)

def wait_running(dec, n, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = metric(dec, "num_running_reqs")
        if v is not None and v >= n: return True
        time.sleep(0.5)
    return False

def wait_idle(dec, timeout=900):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if (metric(dec, "num_running_reqs") or 0) == 0 and (metric(dec, "num_queue_reqs") or 0) == 0: return True
        time.sleep(1)
    return False

def spawn(url, store, key, text, priority=None, max_new_tokens=8, stream=False, timeout=900):
    def w(): store[key] = gen(url, text, priority=priority, max_new_tokens=max_new_tokens, stream=stream, timeout=timeout)
    t = threading.Thread(target=w, daemon=True); t.start(); return t

def holders(url, n, prio, tokens, store, prefix="hold"):
    return [spawn(url, store, f"{prefix}{i}", f"Write a very long story about slot {i}. " * 20, priority=prio, max_new_tokens=tokens, stream=True) for i in range(n)]

def fill(url, n, prio, store, text="Queue filler {k}.", tokens=8, gap=0.25, stream=False):
    ths = []
    for k in range(n):
        ths.append(spawn(url, store, f"fill{k}", text.format(k=k), priority=prio, max_new_tokens=tokens, stream=stream)); time.sleep(gap)
    return ths

def statuses(store, keys): return {k: store.get(k, {}).get("status") for k in keys}

def long_text(path):
    with open(path) as f:
        rec = json.loads(f.readline()); return rec.get("text") or rec.get("prompt")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--decode", required=True)
    ap.add_argument("--mode", required=True, choices=["base", "low", "q0", "q1", "nops", "nolimit", "retract", "soak"])
    ap.add_argument("--slots", type=int, default=8); ap.add_argument("--hold-tokens", type=int, default=9000)
    ap.add_argument("--prompts", default="/home/alisachen_google_com/pd_disagg/prompts/gpt-oss")
    a = ap.parse_args(); U, D, S, H = a.url, a.decode, a.slots, a.hold_tokens
    used0 = metric(D, "num_used_tokens"); say("start", None, f"decode used tokens {used0}, running {metric(D,'num_running_reqs')}")

    def end_check(tag):
        ok = wait_idle(D, 1500); used = metric(D, "num_used_tokens")
        say(f"{tag}-end-idle", ok and used is not None and used < 1024, f"idle={ok} decode num_used_tokens at end = {used}")

    if a.mode == "base":
        # Q1: burst of S holders into an idle decode -> all admitted (no spurious rejection)
        st = {}; th = holders(U, S, 5, 400, st, "b1_")
        for t in th: t.join(1200)
        rej = [k for k in st if st[k]["status"] != 200]
        say("Q1-burst-idle", not rej, f"{S} holders into an idle decode: statuses {sorted(set(v['status'] for v in st.values()))} (rejected: {rej})")
        wait_idle(D)
        # Q2: burst of S+4 into idle -> S run, 4 wait within the limit, none rejected
        st = {}; th = holders(U, S + 4, 5, 400, st, "b2_")
        for t in th: t.join(1200)
        rej = [k for k in st if st[k]["status"] != 200]
        say("Q2-burst-limit-edge", not rej, f"{S+4} holders (limit 4): rejected {rej or 'none'}")
        wait_idle(D)
        # Q3: S running + 4 waiting, then a 13th equal-priority newcomer -> 503
        st = {}; th = holders(U, S, 5, H, st); assert wait_running(D, S), "holders did not all start"
        f = fill(U, 4, 5, st); time.sleep(4)
        r = gen(U, "Newcomer equal.", priority=5, timeout=120); say("Q3-equal-into-full", r["status"] == 503, f"newcomer priority 5 -> HTTP {r['status']} {str(r.get('body'))[:90]}")
        # Q4: lower-priority newcomer -> 503 (D1)
        r = gen(U, "Newcomer low.", priority=3, timeout=120); say("Q4-lower-into-full", r["status"] == 503, f"newcomer priority 3 -> HTTP {r['status']}")
        # Q5: eviction cascade: two higher-priority newcomers -> fillers 3 then 2 evicted, newcomers served
        used_before = metric(D, "num_used_tokens")
        hi = {}; h1 = spawn(U, hi, "hi1", "High newcomer one.", priority=9); time.sleep(3); used_after1 = metric(D, "num_used_tokens")
        h2 = spawn(U, hi, "hi2", "High newcomer two.", priority=9); time.sleep(3)
        for t in f: t.join(1200)
        h1.join(1200); h2.join(1200)
        ev = [k for k in ("fill0", "fill1", "fill2", "fill3") if st[k]["status"] == 503]
        say("Q5-eviction-cascade", ev == ["fill2", "fill3"] and hi["hi1"]["status"] == 200 and hi["hi2"]["status"] == 200, f"evicted {ev} (expect newest two: fill2, fill3); newcomers {hi['hi1']['status']}, {hi['hi2']['status']}; evicted msg: {str(st.get('fill3',{}).get('body'))[:80]}")
        say("Q6-eviction-frees-kv", None, f"decode num_used_tokens before eviction {used_before}, after first eviction {used_after1}")
        for t in th: t.join(1500)
        end_check("Q")
        # Q7: a waiting filler is aborted by its client -> its place frees (how fast through the mini-LB?) and an equal newcomer is admitted
        st = {}; th = holders(U, S, 5, H, st); assert wait_running(D, S)
        f = fill(U, 4, 5, st); time.sleep(3); q_full = metric(D, "num_queue_reqs")
        r_short = gen(U, "Filler that gives up.", priority=5, timeout=4)  # a 5th waiter -> this one gets 503 (queue full); abort one of the queued instead:
        say("Q7a-fifth-waiter", r_short["status"] == 503, f"5th waiter -> HTTP {r_short['status']} (queue {q_full})")
        aborter = {}; ta = spawn(U, aborter, "ab", "Queued then abandoned.", priority=9, timeout=6)  # priority 9 evicts fill3 and takes its place
        ta.join(30)
        say("Q7b-client-abort-waiting", aborter["ab"]["status"] == "ERR", f"priority-9 waiter aborted by its client after 6s: {aborter['ab']['status']}")
        t0 = time.time(); freed = None
        while time.time() - t0 < 90:
            if (metric(D, "num_queue_reqs") or 0) < 4: freed = round(time.time() - t0, 1); break
            time.sleep(1)
        say("Q7c-place-freed", freed is not None, f"waiting count dropped below 4 {freed}s after the client abort (None = not within 90s)")
        r = gen(U, "Newcomer after abort.", priority=5, timeout=900); say("Q7d-admitted-after-abort", r["status"] == 200, f"equal newcomer after the abort -> HTTP {r['status']}")
        for t in f: t.join(1200)
        for t in th: t.join(1500)
        end_check("Q7")
        # Q8: streaming fillers; evicted one must end with an error on its stream
        st = {}; th = holders(U, S, 5, H, st); assert wait_running(D, S)
        f = fill(U, 4, 5, st, stream=True); time.sleep(4)
        r = gen(U, "High newcomer.", priority=9); 
        for t in f: t.join(1200)
        errs = {k: (st[k]["status"], st[k].get("stream_error")) for k in ("fill0","fill1","fill2","fill3")}
        evicted = [k for k, (s, e) in errs.items() if s == 503 or e]
        say("Q8-streaming-filler-evicted", r["status"] == 200 and evicted == ["fill3"], f"newcomer {r['status']}; streaming fillers: {errs}")
        for t in th: t.join(1500)
        end_check("Q8")
        # Q9: long (8k) fillers -> eviction frees a visible amount of KV
        text8k = long_text(f"{a.prompts}/8192.jsonl")
        st = {}; th = holders(U, S, 5, H, st); assert wait_running(D, S)
        f = fill(U, 4, 5, st, text=text8k, gap=1.0); time.sleep(20)
        waiting = metric(D, "num_queue_reqs"); used_before = metric(D, "num_used_tokens")
        r = gen(U, "High newcomer.", priority=9); time.sleep(5); used_after = metric(D, "num_used_tokens")
        for t in f: t.join(1500)
        ev = [k for k in ("fill0","fill1","fill2","fill3") if st[k]["status"] == 503]
        say("Q9-long-filler-evicted", r["status"] == 200 and len(ev) == 1 and used_before is not None and used_after is not None and used_after < used_before, f"waiting before {waiting}; used tokens {used_before} -> {used_after} after evicting {ev} (8k prompt)")
        for t in th: t.join(1500)
        end_check("Q9")
        # Q10: soak - queue kept full; each cycle adds one filler (tops the queue to 4) and one priority-9 newcomer that must evict it
        st = {}; th = holders(U, S, 5, H + 6000, st); assert wait_running(D, S)
        f = fill(U, 3, 5, st); time.sleep(3)
        cyc = []; fs = {}; hs = {}
        for c in range(6):
            tf = spawn(U, fs, f"f{c}", f"Cycle filler {c}.", priority=5); time.sleep(1.5)
            r = gen(U, f"High newcomer {c}.", priority=9); tf.join(30)
            cyc.append((r["status"], fs.get(f"f{c}", {}).get("status"), metric(D, "num_queue_reqs")))
        ok = all(s_new == 200 and s_f == 503 for s_new, s_f, _ in cyc)
        say("Q10-soak-cycles", ok, f"6 cycles of (newcomer, that cycle's filler, waiting after): {cyc}")
        for t in f: t.join(1500)
        for t in th: t.join(1800)
        end_check("Q10")
        say("exceptions", None, "see phase log")

    elif a.mode == "soak":
        # Queue kept full; six newcomers of rising priority (9..14) each evict exactly one waiter:
        # the newest priority-5 filler first (fill3, fill2, fill1, fill0), then the oldest newcomers (9, 10).
        st = {}; th = holders(U, S, 5, H + 6000, st); assert wait_running(D, S), "holders did not all start"
        f = fill(U, 4, 5, st); time.sleep(4)
        expected = ["fill3", "fill2", "fill1", "fill0", "n0", "n1"]; obs = []
        for c in range(6):
            spawn(U, st, f"n{c}", f"Soak newcomer {c}.", priority=9 + c); time.sleep(4)
            victims = [k for k in list(st) if (k.startswith("fill") or k.startswith("n")) and st[k].get("status") == 503]
            obs.append(victims[-1] if len(victims) == c + 1 else f"got {victims}")
        say("S1-soak-evictions", obs == expected, f"victim per cycle: {obs} (expected {expected})")
        for t in th: t.join(1800)
        time.sleep(5)
        survivors = {k: st.get(k, {}).get("status") for k in ("n2", "n3", "n4", "n5")}
        say("S2-survivors-served", all(v == 200 for v in survivors.values()), f"remaining newcomers (priority 11-14) served after the holders: {survivors}")
        end_check("S")

    elif a.mode == "low":
        st = {}; th = holders(U, S, 5, H, st); assert wait_running(D, S)
        f = fill(U, 4, 5, st); time.sleep(4)
        r = gen(U, "Newcomer 9 (worse under low-first).", priority=9, timeout=120); say("L1-higher-value-rejected", r["status"] == 503, f"low-values-first: newcomer priority 9 -> HTTP {r['status']}")
        hi = gen(U, "Newcomer 1 (better under low-first).", priority=1)
        for t in f: t.join(1200)
        ev = [k for k in ("fill0","fill1","fill2","fill3") if st[k]["status"] == 503]
        say("L2-lower-value-evicts", hi["status"] == 200 and ev == ["fill3"], f"newcomer priority 1 -> {hi['status']}, evicted {ev}")
        for t in th: t.join(1500)
        end_check("L")

    elif a.mode == "q0":
        st = {}; th = holders(U, S, 5, 400, st, "z1_")
        for t in th: t.join(1200)
        say("Z1-burst-into-idle", all(v["status"] == 200 for v in st.values()), f"limit 0: {S} holders into idle -> {sorted(set(v['status'] for v in st.values()))}")
        wait_idle(D)
        st = {}; th = holders(U, S, 5, H, st); assert wait_running(D, S)
        r = gen(U, "Any newcomer.", priority=5, timeout=120); say("Z2-any-waiter-rejected", r["status"] == 503, f"limit 0, slots full: newcomer -> HTTP {r['status']}")
        r = gen(U, "High newcomer.", priority=9, timeout=120); say("Z3-nothing-to-evict", r["status"] == 503, f"limit 0: priority 9 with an empty queue -> HTTP {r['status']} (no waiter to evict)")
        for t in th: t.join(1500)
        end_check("Z")

    elif a.mode == "q1":
        st = {}; th = holders(U, S, 5, H, st); assert wait_running(D, S)
        f = fill(U, 1, 5, st); time.sleep(3)
        r = gen(U, "Second waiter.", priority=5, timeout=120); say("O1-second-waiter-rejected", r["status"] == 503, f"limit 1: second equal waiter -> HTTP {r['status']}")
        hi = gen(U, "High newcomer.", priority=9)
        for t in f: t.join(1200)
        say("O2-single-waiter-evicted", hi["status"] == 200 and st["fill0"]["status"] == 503, f"priority 9 -> {hi['status']}, the single waiter -> {st['fill0']['status']}")
        for t in th: t.join(1500)
        end_check("O")

    elif a.mode == "nops":
        st = {}; th = holders(U, S, None, H, st); assert wait_running(D, S)
        f = fill(U, 4, None, st); time.sleep(4)
        r = gen(U, "Fifth waiter.", timeout=120); say("N1-limit-without-priority", r["status"] == 503, f"priority scheduling off, limit 4: 5th waiter -> HTTP {r['status']}")
        for t in f: t.join(1200)
        say("N2-no-eviction", all(st[k]["status"] == 200 for k in ("fill0","fill1","fill2","fill3")), f"fillers all served (no eviction possible): {statuses(st, ['fill0','fill1','fill2','fill3'])}")
        for t in th: t.join(1500)
        end_check("N")

    elif a.mode == "nolimit":
        st = {}; th = holders(U, S + 5, 5, H, st)
        time.sleep(8); peak = metric(D, "num_queue_reqs")
        for t in th: t.join(1500)
        say("U1-no-limit-admits-all", all(v["status"] == 200 for v in st.values()), f"{S+5} holders, no limit: statuses {sorted(set(v['status'] for v in st.values()))}, peak waiting {peak}")
        end_check("U")

    elif a.mode == "retract":
        text131k = long_text(f"{a.prompts}/131072.jsonl")
        st = {}; th = [spawn(U, st, f"long{i}", text131k, priority=5, max_new_tokens=1024, stream=True, timeout=1500) for i in range(S)]
        time.sleep(60); retr0 = metric(D, "num_retracted_reqs")
        f = fill(U, 4, 5, st); time.sleep(5)
        r = gen(U, "High newcomer during pressure.", priority=9, timeout=900)
        for t in f: t.join(1500)
        for t in th: t.join(1800)
        longs = statuses(st, [f"long{i}" for i in range(S)]); fills = statuses(st, ["fill0","fill1","fill2","fill3"])
        say("R1-long-holders", all(v == 200 for v in longs.values()), f"{S} x 131k holders under a 262k decode pool: {longs}")
        say("R2-fillers-and-newcomer", all(v in (200, 503) for v in fills.values()) and r["status"] in (200, 503), f"fillers {fills}, newcomer {r['status']} (every request answered)")
        say("R3-retractions", None, f"num_retracted_reqs {retr0} -> {metric(D, 'num_retracted_reqs')}")
        end_check("R")

if __name__ == "__main__":
    main()
