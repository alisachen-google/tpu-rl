#!/usr/bin/env python3
"""handshake_timeout_probe.py - boundary cases for PR #1230 (decode handshake bounded by SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT).
Orphan rooms are requests sent straight to the decode engine with bootstrap fields no prefill will ever serve.
  --mode orphans   : B1-B5 on a healthy 8-slot pair (timeout T seconds)
  --mode pressure  : B6 on a small decode pool: requests waiting for KV room longer than T
"""
import argparse, json, random, threading, time
import requests

def post(url, body, stream=False, timeout=900):
    t0 = time.time()
    try:
        r = requests.post(url.rstrip("/") + "/generate", json=body, stream=stream, timeout=timeout)
    except Exception as e:
        return {"status": "ERR", "error": type(e).__name__ + ": " + str(e)[:120], "e2e": round(time.time() - t0, 1)}
    out = {"status": r.status_code, "e2e": round(time.time() - t0, 1), "body": r.text[:260] if r.status_code != 200 else None}
    if stream and r.status_code == 200:
        for line in r.iter_lines():
            if line and b"error" in line: out["stream_error"] = line[:260].decode(errors="replace"); break
        out["e2e"] = round(time.time() - t0, 1)
    return out

def orphan(decode_url, prefill_host, bport, text="Orphan room.", tokens=8, timeout=900, host=None):
    body = {"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": tokens, "ignore_eos": True},
            "bootstrap_host": host or prefill_host, "bootstrap_port": bport, "bootstrap_room": random.getrandbits(62)}
    return post(decode_url, body, timeout=timeout)

def metric(dec, name, must='tp_rank="0"'):
    try: txt = requests.get(dec.rstrip("/") + "/metrics", timeout=10).text
    except Exception: return None
    vals = [l for l in txt.splitlines() if l.startswith(f"sglang:{name}{{") and (must is None or must in l)]
    return float(vals[0].rsplit(" ", 1)[1]) if vals else None

def say(case, ok, msg): print(f"[{case}] {'PASS' if ok is True else 'FAIL' if ok is False else 'INFO'} {msg}", flush=True)
def health(u):
    try: return requests.get(u.rstrip("/") + "/health", timeout=5).status_code
    except Exception: return "ERR"
def long_text(path):
    with open(path) as f: rec = json.loads(f.readline()); return rec.get("text") or rec.get("prompt")
MESSAGE = "KV metadata"  # overridden by --message for the waiting-timeout window
def timed_out(r): return r["status"] == 500 and r.get("body") and "timed out" in r["body"] and MESSAGE in r["body"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lb", required=True); ap.add_argument("--decode", required=True); ap.add_argument("--prefill", required=True)
    ap.add_argument("--prefill-host", required=True); ap.add_argument("--bootstrap-port", type=int, default=8998)
    ap.add_argument("--timeout", type=float, required=True, help="SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT the pair runs with")
    ap.add_argument("--message", default="KV metadata", help="substring the timeout error must carry")
    ap.add_argument("--mode", required=True, choices=["orphans", "pressure"]); ap.add_argument("--prompts", default="/home/alisachen_google_com/pd_disagg/prompts/gpt-oss")
    a = ap.parse_args(); LB, D, P, T = a.lb, a.decode, a.prefill, a.timeout
    global MESSAGE; MESSAGE = a.message
    say("start", None, f"timeout {T}s; decode used {metric(D,'num_used_tokens')} prealloc {metric(D,'num_decode_prealloc_queue_reqs')}")

    if a.mode == "orphans":
        # B1: one orphan room -> fails about T seconds later with the new message, not a hang
        r = orphan(D, a.prefill_host, a.bootstrap_port, timeout=T * 3 + 60)
        say("B1-orphan-times-out", timed_out(r) and T - 5 <= r["e2e"] <= T + 30, f"status {r['status']} after {r['e2e']}s; body: {str(r.get('body') or r.get('error'))[:200]}")
        time.sleep(3)
        say("B1-decode-clean", (metric(D, "num_decode_prealloc_queue_reqs") or 0) == 0 and health(D) == 200, f"prealloc queue {metric(D,'num_decode_prealloc_queue_reqs')}, health {health(D)}")
        # B2: 16 concurrent orphans -> all fail around T, decode stays healthy, nothing left behind
        res = {}; ths = []
        for i in range(16):
            def w(i=i): res[i] = orphan(D, a.prefill_host, a.bootstrap_port, text=f"Orphan {i}.", timeout=T * 3 + 60)
            t = threading.Thread(target=w, daemon=True); t.start(); ths.append(t)
        time.sleep(5); inq = metric(D, "num_decode_prealloc_queue_reqs")
        for t in ths: t.join(T * 3 + 90)
        ok = all(timed_out(r) for r in res.values()); e2es = sorted(round(r["e2e"]) for r in res.values())
        say("B2-16-orphans", ok and e2es[-1] <= T + 45, f"in prealloc queue while waiting: {inq}; e2e spread {e2es[0]}..{e2es[-1]}s; statuses {sorted(set(str(r['status']) for r in res.values()))}")
        time.sleep(3)
        say("B2-decode-clean", (metric(D, "num_decode_prealloc_queue_reqs") or 0) == 0 and (metric(D, "num_used_tokens") or 0) < 1024 and health(D) == 200, f"prealloc {metric(D,'num_decode_prealloc_queue_reqs')}, used {metric(D,'num_used_tokens')}, health {health(D)}")
        # B3: unreachable bootstrap host -> bounded too (how, and how fast)
        r = orphan(D, a.prefill_host, a.bootstrap_port, host="10.255.255.1", timeout=T * 3 + 60)
        say("B3-unreachable-host", r["status"] != "ERR" and r["e2e"] <= T + 45, f"status {r['status']} after {r['e2e']}s; body: {str(r.get('body') or r.get('error'))[:200]}")
        # B4: client gives up before the timeout; the timer must not break anything when it later fires
        r = orphan(D, a.prefill_host, a.bootstrap_port, text="Impatient orphan.", timeout=10)
        say("B4a-client-abort-early", r["status"] == "ERR", f"client timed out at 10s: {r['status']} {str(r.get('error'))[:60]}")
        time.sleep(T + 15)
        say("B4b-clean-after-deadline", (metric(D, "num_decode_prealloc_queue_reqs") or 0) == 0 and health(D) == 200, f"{int(T)+15}s later: prealloc {metric(D,'num_decode_prealloc_queue_reqs')}, health {health(D)} (exceptions counted by the phase)")
        # B5: normal traffic through the LB is unaffected: 8k x4 c=4, 32k, 131k
        t8 = long_text(f"{a.prompts}/8192.jsonl"); t32 = long_text(f"{a.prompts}/32768.jsonl"); t131 = long_text(f"{a.prompts}/131072.jsonl")
        res = {}; ths = []
        for i in range(4):
            def w(i=i): res[f"8k{i}"] = post(LB, {"text": t8, "sampling_params": {"temperature": 0, "max_new_tokens": 64, "ignore_eos": True}})
            t = threading.Thread(target=w, daemon=True); t.start(); ths.append(t)
        for t in ths: t.join(900)
        res["32k"] = post(LB, {"text": t32, "sampling_params": {"temperature": 0, "max_new_tokens": 64, "ignore_eos": True}})
        res["131k"] = post(LB, {"text": t131, "sampling_params": {"temperature": 0, "max_new_tokens": 64, "ignore_eos": True}})
        say("B5-normal-traffic", all(r["status"] == 200 for r in res.values()), f"statuses {{k: (r['status'], r['e2e']) for k, r in res.items()}}".replace("{{","{").replace("}}","}") if False else f"statuses { {k: (r['status'], r['e2e']) for k, r in res.items()} }")
        say("B5-idle", (metric(D, "num_used_tokens") or 0) < 1024 and health(D) == 200 and health(P) == 200, f"decode used {metric(D,'num_used_tokens')}, health P {health(P)} D {health(D)}")

    elif a.mode == "pressure":
        # B6: decode pool 262k, 8 x 131k prompts with long outputs at c=8 -> later requests wait for KV room; does the handshake deadline hit them?
        t131 = long_text(f"{a.prompts}/131072.jsonl"); res = {}; ths = []
        for i in range(8):
            def w(i=i): res[i] = post(LB, {"text": t131, "sampling_params": {"temperature": 0, "max_new_tokens": 2048, "ignore_eos": True}}, timeout=1800)
            t = threading.Thread(target=w, daemon=True); t.start(); ths.append(t); time.sleep(0.3)
        time.sleep(20); say("B6-pressure-snapshot", None, f"at +20s: running {metric(D,'num_running_reqs')}, prealloc queue {metric(D,'num_decode_prealloc_queue_reqs')}, transfer {metric(D,'num_decode_transfer_queue_reqs')}, used {metric(D,'num_used_tokens')}")
        for t in ths: t.join(1800)
        st = {i: (r["status"], r["e2e"]) for i, r in sorted(res.items())}
        n_to = sum(1 for r in res.values() if timed_out(r)); n_ok = sum(1 for r in res.values() if r["status"] == 200)
        say("B6-kv-pressure-vs-deadline", None, f"8 x 131k/2048 under a 262k pool, timeout {T}s: ok {n_ok}, handshake-timeouts {n_to}, other {8-n_ok-n_to}; {st}")
        say("B6-verdict", n_to == 0, "no legitimately waiting request was timed out" if n_to == 0 else f"{n_to} requests waiting for KV room were failed by the handshake deadline (regression risk: the window includes decode-side pre-allocation waits)")
        time.sleep(3); say("B6-idle", (metric(D, "num_used_tokens") or 0) < 1024 and health(D) == 200 and health(P) == 200, f"decode used {metric(D,'num_used_tokens')}, health P {health(P)} D {health(D)}")

if __name__ == "__main__":
    main()
