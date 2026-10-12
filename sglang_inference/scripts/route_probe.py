#!/usr/bin/env python3
"""route_probe.py - 2P/2D routing checks through the router. One verdict line per case: [case] PASS|FAIL|INFO ..."""
import argparse, json, re, threading, time, requests

def metric(url, name, mode=None):
    try: txt = requests.get(url.rstrip("/") + "/metrics", timeout=10).text
    except Exception: return None
    tot = 0.0; found = False
    for line in txt.splitlines():
        if line.startswith("sglang:" + name + "{") and ('tp_rank=' not in line or 'tp_rank="0"' in line) and (mode is None or f'mode="{mode}"' in line):
            try: tot += float(line.rsplit(" ", 1)[1]); found = True
            except ValueError: pass
    return tot if found else None

def gen(url, text, max_new=32, rid=None, timeout=300):
    body = {"text": text, "sampling_params": {"max_new_tokens": max_new, "temperature": 0, "ignore_eos": True}}
    if rid: body["rid"] = rid
    t0 = time.time()
    try:
        r = requests.post(url.rstrip("/") + "/generate", json=body, timeout=timeout)
        out = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        return {"status": r.status_code, "text": out.get("text"), "ids": out.get("output_ids"), "e2e": round(time.time() - t0, 2)}
    except Exception as e:
        return {"status": "ERR", "err": str(e)[:80], "e2e": round(time.time() - t0, 2)}

def say(case, ok, msg): print(f"[{case}] {'PASS' if ok else 'FAIL' if ok is False else 'INFO'} {msg}", flush=True)

ap = argparse.ArgumentParser()
ap.add_argument("--router", required=True); ap.add_argument("--prefills", required=True, help="comma list of http://host:port"); ap.add_argument("--decodes", required=True)
ap.add_argument("--n", type=int, default=24); ap.add_argument("--concurrency", type=int, default=4)
a = ap.parse_args(); P = a.prefills.split(","); D = a.decodes.split(",")

def snap():
    return {u: (metric(u, "prompt_tokens_total"), metric(u, "generation_tokens_total"), metric(u, "num_requests_total")) for u in P + D}

say("start", None, f"router {a.router}; prefills {P}; decodes {D}")
for u in P + D:
    say("health", requests.get(u + "/health", timeout=10).status_code == 200, u)
before = snap()
# R1: N distinct prompts spread over all engines
prompts = [f"Routing probe request {i}: list three facts about the number {i*7+1}." for i in range(a.n)]
res = [None] * a.n
def worker(idx):
    res[idx] = gen(a.router, prompts[idx], 32, rid=f"route-{idx}")
ths = []
for i in range(a.n):
    t = threading.Thread(target=worker, args=(i,)); t.start(); ths.append(t)
    if len(ths) % a.concurrency == 0: [x.join() for x in ths]; ths = []
[x.join() for x in ths]
codes = [r["status"] for r in res]
say("R1-all-served", all(c == 200 for c in codes), f"{a.n} requests at c={a.concurrency}: statuses {sorted(set(map(str, codes)))}, e2e {min(r['e2e'] for r in res)}..{max(r['e2e'] for r in res)}s")
time.sleep(3); after = snap()
delta = {u: tuple((after[u][k] or 0) - (before[u][k] or 0) for k in range(3)) for u in P + D}
pused = [u for u in P if delta[u][0] > 0 or delta[u][2] > 0]; dused = [u for u in D if delta[u][1] > 0 or delta[u][2] > 0]
say("R2-all-prefills-used", len(pused) == len(P), f"prefill deltas (prompt_tokens, gen_tokens, requests): { {u: delta[u] for u in P} }")
say("R3-all-decodes-used", len(dused) == len(D), f"decode deltas: { {u: delta[u] for u in D} }")
# R4: same prompt repeated -> identical greedy output whichever pair served it
same = [gen(a.router, "Explain in one sentence why the sky is blue.", 48, rid=f"same-{i}") for i in range(8)]
ids = [tuple(r["ids"] or []) for r in same]
say("R4-pair-invariant-output", all(r["status"] == 200 for r in same) and len(set(ids)) == 1, f"{len(set(ids))} distinct outputs over 8 repeats; statuses {sorted(set(map(str,[r['status'] for r in same])))}")
time.sleep(5)
idle = all((metric(u, "num_running_reqs") or 0) == 0 for u in D)
say("R5-idle-after", idle, f"decode running: { {u: metric(u,'num_running_reqs') for u in D} }")
