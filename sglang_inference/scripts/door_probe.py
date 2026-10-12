#!/usr/bin/env python3
"""door_probe.py - live cases for PD door admission (#1299). One line per case: [case] PASS|FAIL|INFO.
Modes: door (limit 4, priority on, 8 slots), q0, q1, nops, low, retract.
Checks: admission math, 503 status/latency through the router, no prefill work for refused requests,
bursts, client-abort storms, streaming/long-prompt refusals, soak cycles, greedy-output equality."""
import argparse, sys, threading, time, requests
sys.path.insert(0, "/home/alisachen_google_com/pd_disagg/scripts")
from queue_limit_probe import gen, metric, say, wait_running, wait_idle, spawn, holders, fill, long_text  # noqa: E402

def gen_rid(url, text, rid, max_new_tokens=8, priority=None, stream=False, timeout=120):
    body = {"text": text, "rid": rid, "sampling_params": {"temperature": 0, "max_new_tokens": max_new_tokens, "ignore_eos": True}}
    if priority is not None: body["priority"] = priority
    t0 = time.time()
    try:
        r = requests.post(url.rstrip("/") + "/generate", json=body, stream=stream, timeout=timeout)
        if stream:
            first = next(r.iter_lines(), b"")
            return {"status": r.status_code, "body": first[:200].decode(errors="replace"), "e2e": time.time() - t0}
        return {"status": r.status_code, "body": r.text[:300], "e2e": time.time() - t0}
    except Exception as e:
        return {"status": "ERR", "body": str(e)[:160], "e2e": time.time() - t0}

def is_refused(r): return r["status"] == 503 and "queue is full" in (r.get("body") or "").lower()

def end_check(tag, D, P):
    ok = wait_idle(D, 900); say(f"{tag}-end-idle", ok, f"decode idle={ok} used_tokens={metric(D,'num_used_tokens')} queue={metric(D,'num_queue_reqs')}")
    pb = metric(P, "num_prefill_bootstrap_queue_reqs"); pq = metric(P, "num_queue_reqs"); pr = metric(P, "num_running_reqs")
    say(f"{tag}-prefill-clean", (pb or 0) == 0 and (pq or 0) == 0 and (pr or 0) == 0, f"prefill bootstrap_queue={pb} queue={pq} running={pr}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--decode", required=True); ap.add_argument("--prefill", required=True)
    ap.add_argument("--mode", required=True, choices=["door", "q0", "q1", "nops", "low", "retract"]); ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--limit", type=int, default=4); ap.add_argument("--hold-tokens", type=int, default=6000)
    ap.add_argument("--prompts", default="/home/alisachen_google_com/pd_disagg/prompts/gpt-oss")
    a = ap.parse_args(); U, D, P, S, L, H = a.url, a.decode, a.prefill, a.slots, a.limit, a.hold_tokens
    prio = None if a.mode == "nops" else 5
    say("start", None, f"mode={a.mode} decode used={metric(D,'num_used_tokens')} prefill prompt_tokens_total={metric(P,'prompt_tokens_total')}")

    if a.mode in ("door", "nops", "low"):
        tag = {"door": "D", "nops": "N", "low": "L"}[a.mode]
        # 1. idle burst of 20: slots + limit admitted, the rest refused quickly, nothing errors or hangs
        st = {}; th = [spawn(U, st, f"b{i}", f"Burst request {i}. " * 4, priority=prio, max_new_tokens=600, stream=True) for i in range(20)]
        for t in th: t.join(600)
        codes = [st[f"b{i}"]["status"] for i in range(20)]; n200, n503 = codes.count(200), codes.count(503)
        say(f"{tag}1-idle-burst", n200 == S + L and n503 == 20 - S - L and "ERR" not in codes, f"20 into idle (slots {S}, limit {L}): 200×{n200} 503×{n503} other={[c for c in codes if c not in (200,503)]}")
        end_check(f"{tag}1", D, P)
        # 2. slots full + queue full: refused newcomer gets 503 with the reason, fast, through the router
        st = {}; th = holders(U, S, prio, H, st); assert wait_running(D, S), "holders did not all start"
        f = fill(U, L, prio, st, tokens=64); time.sleep(4)
        pt0 = metric(P, "prompt_tokens_total")
        r = gen_rid(U, "Door newcomer equal priority.", "door-eq", priority=prio)
        say(f"{tag}2-refused-503", is_refused(r) and r["e2e"] < 5, f"newcomer -> HTTP {r['status']} in {r['e2e']:.2f}s; body: {r['body'][:120]}")
        if a.mode != "nops":
            r2 = gen_rid(U, "Door newcomer higher priority.", "door-hi", priority=(1 if a.mode == "low" else 9))
            say(f"{tag}3-no-eviction", is_refused(r2), f"higher-priority newcomer (door v1 refuses, no eviction) -> HTTP {r2['status']} in {r2['e2e']:.2f}s")
        # 3. refused requests cost the prefill nothing: 3 × 8k prompts, prefill prompt_tokens_total unchanged
        big = long_text(f"{a.prompts}/8192.jsonl")
        rs = [gen_rid(U, big, f"door-big-{i}", priority=prio) for i in range(3)]
        time.sleep(5); pt1 = metric(P, "prompt_tokens_total")
        say(f"{tag}4-no-prefill-work", all(is_refused(x) for x in rs) and pt1 == pt0, f"3 refused 8k prompts: statuses {[x['status'] for x in rs]} e2e {[round(x['e2e'],2) for x in rs]}; prefill prompt_tokens_total {pt0} -> {pt1}")
        # 4. streaming refusal
        r = gen_rid(U, "Streaming newcomer.", "door-stream", priority=prio, stream=True)
        say(f"{tag}5-stream-refused", r["status"] == 503, f"streaming newcomer -> HTTP {r['status']} first line: {r['body'][:100]}")
        for t in f: t.join(900)
        # 5. abort storm while full: 16 newcomers whose clients give up after 1.5 s
        storm = {}; sth = [threading.Thread(target=lambda i=i: storm.__setitem__(i, gen_rid(U, f"Storm {i}.", f"door-storm-{i}", priority=prio, timeout=1.5)), daemon=True) for i in range(16)]
        [t.start() for t in sth]; [t.join(10) for t in sth]
        kinds = {}
        for v in storm.values(): kinds[v["status"]] = kinds.get(v["status"], 0) + 1
        say(f"{tag}6-abort-storm", True, f"16 newcomers with 1.5s client timeout: {kinds}")
        for t in th: t.join(1500)
        end_check(f"{tag}6", D, P)
        r = gen_rid(U, "After the storm.", "door-after", priority=prio); say(f"{tag}7-serves-after-storm", r["status"] == 200, f"plain request after storm -> HTTP {r['status']}")
        # 6. soak: 5 cycles of fill-to-limit + 3 refusals + drain; counts and idleness stable
        cyc = []
        for c in range(5):
            st = {}; th = holders(U, S, prio, 2500, st); wait_running(D, S); f = fill(U, L, prio, st, tokens=32); time.sleep(3)
            ref = [gen_rid(U, f"Soak {c} newcomer {k}.", f"door-soak-{c}-{k}", priority=prio) for k in range(3)]
            for t in f: t.join(600)
            for t in th: t.join(900)
            idle = wait_idle(D, 600); cyc.append((sum(is_refused(x) for x in ref), sum(1 for k in st if st[k]["status"] == 200), idle, metric(P, "num_prefill_bootstrap_queue_reqs")))
        say(f"{tag}8-soak-cycles", all(c == (3, S + L, True, 0) for c in cyc), f"per cycle (refused, served, idle, prefill bootstrap queue): {cyc} expected (3, {S+L}, True, 0)")
        # 7. greedy outputs unchanged under pressure
        probes = ["The capital of France is", "List three prime numbers:", "Water boils at"]
        base = [gen_rid(U, p, f"door-base-{i}", max_new_tokens=16, priority=prio)["body"] for i, p in enumerate(probes)]
        st = {}; th = holders(U, S, prio, 2500, st); wait_running(D, S); time.sleep(1)
        under = [gen_rid(U, p, f"door-under-{i}", max_new_tokens=16, priority=prio, timeout=600)["body"] for i, p in enumerate(probes)]
        for t in th: t.join(900)
        import json as _j
        def txt(b):
            try: return _j.loads(b).get("text")
            except Exception: return b[:60]
        same = [txt(x) == txt(y) for x, y in zip(base, under)]
        say(f"{tag}9-greedy-equal-under-pressure", all(same), f"3 greedy prompts idle vs admitted-under-pressure equal: {same}")
        end_check(f"{tag}9", D, P)

    elif a.mode in ("q0", "q1"):
        tag = "Z" if a.mode == "q0" else "O"
        st = {}; th = [spawn(U, st, f"b{i}", f"Burst {i}. " * 4, priority=prio, max_new_tokens=400, stream=True) for i in range(S + 4)]
        for t in th: t.join(600)
        codes = [st[f"b{i}"]["status"] for i in range(S + 4)]
        say(f"{tag}1-idle-burst", codes.count(200) == S + L and codes.count(503) == 4 - L and "ERR" not in codes, f"{S+4} into idle, limit {L}: 200×{codes.count(200)} 503×{codes.count(503)} other={[c for c in codes if c not in (200,503)]}")
        end_check(f"{tag}1", D, P)
        st = {}; th = holders(U, S, prio, H, st); assert wait_running(D, S)
        f = fill(U, L, prio, st, tokens=64) if L else []; time.sleep(3)
        r = gen_rid(U, "One too many.", "door-over", priority=prio); say(f"{tag}2-over-limit-503", is_refused(r) and r["e2e"] < 5, f"limit {L}: waiter #{L+1} -> HTTP {r['status']} in {r['e2e']:.2f}s")
        r = gen_rid(U, "Priority 9 with nothing to evict.", "door-p9", priority=9); say(f"{tag}3-priority-no-crash", is_refused(r), f"priority 9 newcomer -> HTTP {r['status']} (no eviction; scheduler alive)")
        for t in f: t.join(900)
        for t in th: t.join(1500)
        end_check(f"{tag}3", D, P)

    elif a.mode == "retract":
        # small decode pool: long holders force retractions; the door counts retracted requests as occupying slots
        st = {}; th = holders(U, S, prio, 20000, st); assert wait_running(D, S)
        t0 = time.time(); seen_retract = 0; refused = []
        while time.time() - t0 < 240:
            seen_retract = max(seen_retract, metric(D, "num_retracted_reqs") or 0)
            if seen_retract: break
            time.sleep(2)
        say("R1-retraction-observed", seen_retract > 0, f"decode num_retracted_reqs peaked at {seen_retract} within {time.time()-t0:.0f}s")
        for k in range(3):
            refused.append(gen_rid(U, f"Under retraction {k}.", f"door-retract-{k}", priority=prio)); time.sleep(1)
        say("R2-newcomers-under-retraction", all(x["status"] in (200, 503) for x in refused), f"newcomers while retracted: {[x['status'] for x in refused]} (503 when retracted+waiting exceed the limit)")
        for t in th: t.join(2400)
        say("R3-holders-finish", all(st[f"hold{i}"]["status"] == 200 for i in range(S)), f"holders statuses {[st[f'hold{i}']['status'] for i in range(S)]} (no deadlock)")
        end_check("R3", D, P)

if __name__ == "__main__":
    main()
