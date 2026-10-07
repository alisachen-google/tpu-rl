#!/usr/bin/env python3
"""abort_test.py - client-abort robustness for SGLang PD disaggregation.

  python3 abort_test.py --url http://router:port --prompt-file prompts.jsonl [--prompt-id tiny8192] \
      [--phase prefill|decode] [--abort-after 10] [--repeat 5] [--concurrent 1] \
      [--server-info http://prefill:port --server-info http://decode:port] [--settle 60] \
      [--max-new-tokens 4096] [--no-ignore-eos] [--timeout 300] [--report abort_report.json]

Each aborting request is a streaming /generate (temperature 0, ignore_eos unless --no-ignore-eos so
decode runs long enough to abort).  The abort is a hard close of the streaming HTTP connection
(socket shutdown + resp.close()).  Phase `prefill` closes while still waiting for the first
streamed token (at min(--abort-after, time-of-first-token)); phase `decode` closes --abort-after
seconds after the first token.  --repeat rounds of --concurrent simultaneous aborts are run.
/get_server_info on every --server-info URL is polled before, right after the aborts and again
after --settle seconds; num_running_reqs / num_queue_reqs / token_usage must return to baseline
(running == 0, queue == 0, |token_usage - baseline| <= 0.02).  Then one short normal request and
one request identical to the aborted prompt (max_new_tokens 32) must succeed; the latter's
cached_tokens is reported.  Exit 1 if resources do not settle, follow-ups fail, or an aborting
request could not be started.
"""
import argparse
import json
import socket
import sys
import threading
import time

import requests

T0 = time.time()
LOCK = threading.Lock()
TIMELINE = []


def log(msg):
    t = time.time() - T0
    with LOCK:
        TIMELINE.append({"t": round(t, 3), "msg": msg})
        print(f"[+{t:8.2f}s] {msg}", flush=True)


def load_prompt(path, pid):
    rows = []
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    sys.exit(f"error: {path}:{ln}: invalid JSON: {e}")
    rows = [r for r in rows if isinstance(r.get("text"), str) and r["text"]]
    if not rows:
        sys.exit(f"error: no non-empty 'text' prompts in {path}")
    if pid is not None:
        rows = [r for r in rows if str(r.get("id")) == pid]
        if not rows:
            sys.exit(f"error: prompt id {pid!r} not found in {path}")
        return rows[0]
    return max(rows, key=lambda r: r.get("isl") or len(r["text"]))


def server_info(url, timeout=30):
    """Return {num_running_reqs, num_queue_reqs, token_usage} (max over DP ranks) or {'error':...}."""
    try:
        r = requests.get(url.rstrip("/") + "/get_server_info", timeout=timeout)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:120]}"}
        j = r.json()
    except (requests.RequestException, ValueError) as e:
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    states = [j] + [s for s in (j.get("internal_states") or []) if isinstance(s, dict)]
    out = {}
    for k in ("num_running_reqs", "num_queue_reqs", "token_usage"):
        vals = [s[k] for s in states if isinstance(s.get(k), (int, float))]
        out[k] = max(vals) if vals else None
    return out


def poll_all(urls, label):
    snap = {}
    for u in urls:
        snap[u] = server_info(u)
        log(f"server-info[{label}] {u}: {snap[u]}")
    return snap


def hard_close(resp):
    """Abort by shutting down the underlying socket, then releasing the response."""
    for getter in (lambda: resp.raw._connection.sock, lambda: resp.raw._fp.fp.raw._sock):
        try:
            sock = getter()
            if sock is not None:
                sock.shutdown(socket.SHUT_RDWR)
                sock.close()
                break
        except Exception:  # noqa: BLE001 - best effort across urllib3/http.client internals
            continue
    try:
        resp.close()
    except Exception:  # noqa: BLE001
        pass


def abort_one(idx, text, args):
    rec = {"idx": idx, "rid": f"abort-{idx}-{int(time.time() * 1000)}", "status": None, "error": None,
           "first_token_s": None, "aborted_s": None, "chunks": 0, "phase_actual": None, "finished": False}
    body = {"text": text, "stream": True, "rid": rec["rid"],
            "sampling_params": {"temperature": 0, "max_new_tokens": args.max_new_tokens,
                                "ignore_eos": not args.no_ignore_eos}}
    first, done = threading.Event(), threading.Event()
    t_send = time.time()
    try:
        resp = requests.post(args.url.rstrip("/") + "/generate", json=body, stream=True,
                             timeout=(10, args.timeout))
    except requests.RequestException as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        log(f"req{idx}: could not start: {rec['error']}")
        return rec
    rec["status"] = resp.status_code
    if resp.status_code != 200:
        rec["error"] = f"HTTP {resp.status_code}: {resp.text[:200]}"
        resp.close()
        log(f"req{idx}: {rec['error']}")
        return rec
    log(f"req{idx}: sent ({rec['rid']}), headers after {time.time() - t_send:.2f}s")

    def reader():
        try:
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                payload = raw[5:].strip()
                if payload == "[DONE]":
                    rec["finished"] = True
                    break
                chunk = json.loads(payload)
                if chunk.get("text") or (chunk.get("meta_info") or {}).get("completion_tokens"):
                    rec["chunks"] += 1
                    if not first.is_set():
                        rec["first_token_s"] = round(time.time() - t_send, 3)
                        first.set()
        except Exception:  # noqa: BLE001 - expected once we close the socket
            pass
        finally:
            done.set()

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    if args.phase == "prefill":
        got = first.wait(args.abort_after)
        rec["phase_actual"] = "decode (first token beat the abort)" if got else "prefill"
    else:
        if not first.wait(args.timeout):
            rec["error"] = f"no first token within {args.timeout}s"
            rec["phase_actual"] = "prefill (never produced a token)"
        elif done.wait(args.abort_after):
            rec["phase_actual"] = "finished before abort" if rec["finished"] else "stream ended early"
        else:
            rec["phase_actual"] = "decode"
    hard_close(resp)
    rec["aborted_s"] = round(time.time() - t_send, 3)
    th.join(5)
    log(f"req{idx}: aborted at {rec['aborted_s']}s in {rec['phase_actual']} "
        f"(first_token={rec['first_token_s']}s, chunks={rec['chunks']})")
    return rec


def follow_up(label, text, mnt, args):
    body = {"text": text, "stream": False, "rid": f"followup-{label}-{int(time.time() * 1000)}",
            "sampling_params": {"temperature": 0, "max_new_tokens": mnt}}
    rec = {"label": label, "ok": False, "status": None, "error": None, "cached_tokens": None,
           "completion_tokens": None, "elapsed": None}
    t = time.time()
    try:
        r = requests.post(args.url.rstrip("/") + "/generate", json=body, timeout=args.timeout)
        rec["status"] = r.status_code
        if r.status_code != 200:
            rec["error"] = r.text[:200]
        else:
            meta = r.json().get("meta_info") or {}
            rec.update({"ok": True, "cached_tokens": meta.get("cached_tokens"),
                        "completion_tokens": meta.get("completion_tokens")})
    except (requests.RequestException, ValueError) as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    rec["elapsed"] = round(time.time() - t, 3)
    log(f"follow-up {label}: {'OK' if rec['ok'] else 'FAIL'} status={rec['status']} "
        f"tokens={rec['completion_tokens']} cached={rec['cached_tokens']} {rec['error'] or ''}")
    return rec


def settled(base, now):
    """Return list of problems comparing a server-info snapshot to its baseline."""
    probs = []
    if "error" in now:
        return [f"poll error: {now['error']}"]
    if now.get("num_running_reqs") is None or now.get("token_usage") is None:
        return []  # fields not exposed: cannot judge, reported as unknown
    if now["num_running_reqs"] != 0:
        probs.append(f"num_running_reqs={now['num_running_reqs']}")
    if (now.get("num_queue_reqs") or 0) != 0:
        probs.append(f"num_queue_reqs={now['num_queue_reqs']}")
    bt = base.get("token_usage")
    if bt is not None and abs(now["token_usage"] - bt) > 0.02:
        probs.append(f"token_usage {now['token_usage']:.3f} vs baseline {bt:.3f}")
    return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="router or server http://host:port")
    ap.add_argument("--prompt-file", required=True, help="jsonl with 'text' (longest used unless --prompt-id)")
    ap.add_argument("--prompt-id", default=None)
    ap.add_argument("--phase", choices=["prefill", "decode"], default="decode")
    ap.add_argument("--abort-after", type=float, default=10)
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--concurrent", type=int, default=1)
    ap.add_argument("--server-info", action="append", default=[], help="engine URL to poll /get_server_info (repeatable)")
    ap.add_argument("--settle", type=float, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=4096, help="for the aborted requests")
    ap.add_argument("--no-ignore-eos", action="store_true")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--report", default="abort_report.json")
    args = ap.parse_args()
    if args.repeat < 1 or args.concurrent < 1:
        sys.exit("error: --repeat and --concurrent must be >= 1")

    prompt = load_prompt(args.prompt_file, args.prompt_id)
    text = prompt["text"]
    log(f"prompt id={prompt.get('id')} isl={prompt.get('isl')} chars={len(text)} phase={args.phase} "
        f"abort_after={args.abort_after}s repeat={args.repeat} concurrent={args.concurrent}")
    baseline = poll_all(args.server_info, "baseline")

    aborts, idx = [], 0
    for rnd in range(1, args.repeat + 1):
        log(f"round {rnd}/{args.repeat}: launching {args.concurrent} aborting request(s)")
        results = [None] * args.concurrent
        ths = []
        for k in range(args.concurrent):
            ths.append(threading.Thread(target=lambda k=k, i=idx + k: results.__setitem__(k, abort_one(i, text, args))))
            ths[-1].start()
        for th in ths:
            th.join()
        idx += args.concurrent
        aborts += results
    after = poll_all(args.server_info, "after-abort")
    if args.server_info:
        log(f"waiting --settle {args.settle}s")
        time.sleep(args.settle)
    settled_snap = poll_all(args.server_info, "settled")

    problems = []
    for u in args.server_info:
        p = settled(baseline[u], settled_snap[u])
        if p:
            problems.append(f"{u}: " + ", ".join(p))
        if settled_snap[u].get("num_running_reqs") is None and "error" not in settled_snap[u]:
            log(f"WARN {u}: server-info lacks num_running_reqs/token_usage; baseline check skipped")
    f1 = follow_up("short", "Hello, my name is", 16, args)
    f2 = follow_up("identical-prompt", text, 32, args)
    final = poll_all(args.server_info, "final")

    start_errors = [a for a in aborts if a["error"] and a["status"] != 200]
    for a in aborts:
        if a["phase_actual"] and a["phase_actual"].startswith(("finished", "decode (first")):
            log(f"WARN req{a['idx']}: abort landed in '{a['phase_actual']}' (tune --abort-after)")
    for p in problems:
        log(f"FAIL not back to baseline: {p}")
    ok = not problems and f1["ok"] and f2["ok"] and not start_errors
    report = {"url": args.url, "prompt_id": prompt.get("id"), "phase": args.phase, "abort_after": args.abort_after,
              "repeat": args.repeat, "concurrent": args.concurrent, "settle": args.settle,
              "server_info": {"baseline": baseline, "after_abort": after, "settled": settled_snap, "final": final},
              "aborts": aborts, "follow_ups": [f1, f2], "baseline_problems": problems,
              "start_errors": len(start_errors), "pass": ok, "timeline": TIMELINE}
    with open(args.report, "w") as f:
        json.dump(report, f, indent=1)
    log(f"{'PASS' if ok else 'FAIL'}: {len(aborts)} aborts ({len(start_errors)} failed to start), "
        f"follow-ups short={f1['ok']} identical={f2['ok']} (cached_tokens={f2['cached_tokens']}), "
        f"{len(problems)} baseline problem(s) -> {args.report}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
