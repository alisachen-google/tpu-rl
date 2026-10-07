#!/usr/bin/env python3
"""pd_watchdog.py - poll SGLang PD servers, tail logs, detect crashes/hangs.

Runtime state (running/queue/token_usage/throughput, PD queue depths) is read from
GET /metrics (Prometheus text; gauges matched by name, max over label sets).
GET /get_server_info is the liveness probe and supplies last_gen_throughput,
memory_usage.token_capacity and any --extra-keys.

Example:
  python3 pd_watchdog.py --server prefill=http://10.0.0.1:30000 \
      --server decode=http://10.0.0.2:30001 --log prefill=/tmp/prefill.log \
      --pid prefill=12345 --interval 10 --duration 3600 --out watchdog.csv

Exit code 0 normally, 2 if any crash or hang alert was raised.
"""
import argparse
import csv
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

import requests

CRASH_SIGS = ["RESOURCE_EXHAUSTED", "Check failed", "Segmentation fault",
              "scheduler died", "Detected recompile", "Watchdog", "Traceback",
              "InductorError", "OutOfMemory", "KeyError 'tpu'"]
COMPILE_SIGS = ["Compiling", "recompile", "cache miss"]
# CSV column -> Prometheus sample name (from sglang/srt/observability/metrics_collector.py)
METRIC_KEYS = {"num_running_reqs": "sglang:num_running_reqs", "num_queue_reqs": "sglang:num_queue_reqs",
               "token_usage": "sglang:token_usage", "gen_throughput": "sglang:gen_throughput",
               "avg_spec_accept_length": "sglang:spec_accept_length",
               "num_used_tokens": "sglang:num_used_tokens", "cache_hit_rate": "sglang:cache_hit_rate"}
# PD-disaggregation gauges/counters (histograms are read via their _count sample)
PD_METRICS = ["sglang:num_prefill_bootstrap_queue_reqs", "sglang:num_prefill_inflight_queue_reqs",
              "sglang:num_decode_prealloc_queue_reqs", "sglang:num_decode_transfer_queue_reqs",
              "sglang:num_decode_host_receive_queue_reqs", "sglang:pending_prealloc_token_usage",
              "sglang:kv_transfer_oldest_inflight_age_seconds", "sglang:num_bootstrap_failed_reqs_total",
              "sglang:num_transfer_failed_reqs_total", "sglang:num_prefill_retries_total",
              "sglang:num_retracted_reqs", "sglang:num_retracted_requests_total",
              "sglang:kv_transfer_latency_ms_count", "sglang:kv_transfer_latency_ms_sum",
              "sglang:kv_transfer_bootstrap_ms_count", "sglang:kv_transfer_bootstrap_ms_sum",
              "sglang:kv_transfer_total_mb_sum"]
PD_COLS = {m.split(":", 1)[1]: m for m in PD_METRICS}
SERVER_INFO_KEYS = ["last_gen_throughput", "token_capacity"]


def parse_prom(text):
    """Parse Prometheus text exposition -> {sample_name: max value over label sets}."""
    out = {}
    for line in text.splitlines():
        if not line or line[0] == "#":
            continue
        name_end = len(line)
        for i, ch in enumerate(line):
            if ch in "{ ":
                name_end = i
                break
        name = line[:name_end]
        rest = line[name_end:]
        if rest.startswith("{"):
            close = rest.rfind("}")
            if close < 0:
                continue
            rest = rest[close + 1:]
        parts = rest.split()
        if not parts:
            continue
        try:
            v = float(parts[0])
        except ValueError:
            continue
        out[name] = v if name not in out or v > out[name] else out[name]
    return out


def parse_kv(items, what):
    """Parse repeatable NAME=VALUE args into a dict; error on malformed input."""
    out = {}
    for it in items or []:
        if "=" not in it:
            sys.exit(f"error: --{what} expects NAME=VALUE, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def lookup(info, key):
    """Find key in internal_states[0] first, then top level. Return '' if absent."""
    states = info.get("internal_states")
    if isinstance(states, list) and states and isinstance(states[0], dict):
        if key in states[0]:
            return states[0][key]
    if key in info:
        return info[key]
    return ""


def to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


class LogTail:
    """Incrementally read a log file, counting crash/compile signatures."""

    def __init__(self, name, path):
        self.name, self.path, self.pos = name, path, 0
        self.crash_hits = []  # (sig, line)
        if os.path.exists(path):
            self.pos = os.path.getsize(path)  # start at end: only new lines

    def poll(self):
        crash, comp = 0, 0
        try:
            size = os.path.getsize(self.path)
            if size < self.pos:  # truncated/rotated
                self.pos = 0
            with open(self.path, "r", errors="replace") as f:
                f.seek(self.pos)
                for line in f:
                    low = line.lower()
                    for s in CRASH_SIGS:
                        if s.lower() in low:
                            crash += 1
                            self.crash_hits.append((s, line.rstrip()))
                            print(f"[{now_iso()}] ALERT crash-sig {s!r} in log "
                                  f"{self.name}: {line.rstrip()[:300]}", flush=True)
                            break
                    for s in COMPILE_SIGS:
                        if s.lower() in low:
                            comp += 1
                            break
                self.pos = f.tell()
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"[{now_iso()}] WARN cannot read log {self.path}: {e}", flush=True)
        return crash, comp


def read_rss_kb(pid):
    """VmRSS in kB from /proc/PID/status (localhost only). None if unavailable."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return None
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", action="append", required=True,
                    help="NAME=URL of an SGLang server (repeatable)")
    ap.add_argument("--log", action="append", help="NAME=PATH log file to tail (repeatable)")
    ap.add_argument("--pid", action="append", help="NAME=PID for RSS tracking via /proc (repeatable)")
    ap.add_argument("--interval", type=float, default=10.0, help="poll interval seconds (default 10)")
    ap.add_argument("--duration", type=float, default=None, help="total seconds (default: until Ctrl-C)")
    ap.add_argument("--hang-secs", type=float, default=300.0,
                    help="alert if running>0 and gen_throughput==0 for this long (default 300)")
    ap.add_argument("--extra-keys", default="", help="comma list of extra keys from internal_states[0]/top level")
    ap.add_argument("--out", default="watchdog.csv", help="CSV output path (default watchdog.csv)")
    args = ap.parse_args()

    servers = parse_kv(args.server, "server")
    logs = {k: LogTail(k, v) for k, v in parse_kv(args.log, "log").items()}
    pids = parse_kv(args.pid, "pid")
    extra = [k for k in args.extra_keys.split(",") if k.strip()]

    fields = (["ts_iso", "server", "http_ok", "latency_ms", "metrics_ok"] + list(METRIC_KEYS)
              + SERVER_INFO_KEYS + extra + list(PD_COLS) + ["log_crash_hits", "log_compile_hits", "rss_kb"])
    new_file = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
    fh = open(args.out, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    if new_file:
        writer.writeheader()

    stats = {"polls": 0, "failures": 0, "alerts": 0, "crash_alerts": 0, "hang_alerts": 0,
             "max_token_usage": {}, "peak_rss_kb": {}}
    hang_since = {}      # server -> ts when stall started
    hang_flagged = set()
    fail_since = {}      # server -> ts when failures started
    fail_flagged = set()
    stop = {"flag": False}

    def on_sig(*_):
        stop["flag"] = True
    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    t0 = time.time()
    sess = requests.Session()
    while not stop["flag"]:
        now = time.time()
        if args.duration is not None and now - t0 >= args.duration:
            break
        ts = now_iso()
        log_counts = {k: t.poll() for k, t in logs.items()}
        for name, url in servers.items():
            stats["polls"] += 1
            row = {"ts_iso": ts, "server": name}
            t1 = time.time()
            info, ok = None, 0
            try:
                r = sess.get(url.rstrip("/") + "/get_server_info", timeout=5)
                row["latency_ms"] = round((time.time() - t1) * 1000, 1)
                if r.status_code == 200:
                    info, ok = r.json(), 1
                else:
                    print(f"[{ts}] WARN {name}: HTTP {r.status_code} from /get_server_info", flush=True)
            except (requests.RequestException, ValueError) as e:
                row["latency_ms"] = round((time.time() - t1) * 1000, 1)
                print(f"[{ts}] WARN {name}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            row["http_ok"] = ok
            # primary runtime state: /metrics (gauges by name, max over labels)
            metrics, m_ok = {}, 0
            try:
                mr = sess.get(url.rstrip("/") + "/metrics", timeout=5)
                if mr.status_code == 200:
                    metrics, m_ok = parse_prom(mr.text), 1
                else:
                    print(f"[{ts}] WARN {name}: HTTP {mr.status_code} from /metrics", flush=True)
            except requests.RequestException as e:
                print(f"[{ts}] WARN {name}: /metrics {type(e).__name__}: {str(e)[:120]}", flush=True)
            row["metrics_ok"] = m_ok
            for col, mname in list(METRIC_KEYS.items()) + list(PD_COLS.items()):
                if mname in metrics:
                    row[col] = metrics[mname]
            if ok:
                fail_since.pop(name, None)
                fail_flagged.discard(name)
                for k in extra:
                    v = lookup(info, k)
                    row[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
                row["last_gen_throughput"] = lookup(info, "last_gen_throughput")
                mem = lookup(info, "memory_usage")
                if isinstance(mem, dict):
                    row["token_capacity"] = mem.get("token_capacity", "")
                if row.get("gen_throughput") is None and row["last_gen_throughput"] != "":
                    row["gen_throughput"] = row["last_gen_throughput"]  # fallback if /metrics absent
            if m_ok or ok:
                tu = to_float(row.get("token_usage"))
                if tu is not None:
                    stats["max_token_usage"][name] = max(stats["max_token_usage"].get(name, 0.0), tu)
                running = to_float(row.get("num_running_reqs")) or 0
                thr = to_float(row.get("gen_throughput"))
                if running > 0 and thr is not None and thr == 0:
                    hang_since.setdefault(name, now)
                    stalled = now - hang_since[name]
                    if stalled > args.hang_secs and name not in hang_flagged:
                        hang_flagged.add(name)
                        stats["alerts"] += 1
                        stats["hang_alerts"] += 1
                        print(f"[{ts}] HANG? {name}: num_running_reqs={running:g} gen_throughput=0 "
                              f"for {stalled:.0f}s (queue={row.get('num_queue_reqs')}, "
                              f"token_usage={row.get('token_usage')}, metrics_ok={m_ok}, "
                              f"pd_queues=" + ",".join(f"{c}={row[c]:g}" for c in PD_COLS
                                                       if "queue" in c and c in row) + ")", flush=True)
                else:
                    hang_since.pop(name, None)
                    hang_flagged.discard(name)
            if not ok:
                stats["failures"] += 1
                fail_since.setdefault(name, now)
                down = now - fail_since[name]
                if down > 60 and name not in fail_flagged:
                    fail_flagged.add(name)
                    stats["alerts"] += 1
                    print(f"[{ts}] ALERT {name}: /get_server_info failing for {down:.0f}s", flush=True)
            if name in log_counts:
                row["log_crash_hits"], row["log_compile_hits"] = log_counts[name]
                if log_counts[name][0]:
                    stats["alerts"] += log_counts[name][0]
                    stats["crash_alerts"] += log_counts[name][0]
            if name in pids:
                rss = read_rss_kb(pids[name])
                if rss is not None:
                    row["rss_kb"] = rss
                    stats["peak_rss_kb"][name] = max(stats["peak_rss_kb"].get(name, 0), rss)
            writer.writerow(row)
            print(f"[{ts}] {name} ok={ok} m={m_ok} lat={row.get('latency_ms')}ms run={row.get('num_running_reqs')} "
                  f"q={row.get('num_queue_reqs')} use={row.get('token_usage')} "
                  f"thr={row.get('gen_throughput')} crash={row.get('log_crash_hits', '-')}", flush=True)
        # count log hits for log names that are not server names
        for lname, (c, _) in log_counts.items():
            if lname not in servers and c:
                stats["alerts"] += c
                stats["crash_alerts"] += c
        fh.flush()
        # sleep in small steps so Ctrl-C / duration are responsive
        deadline = now + args.interval
        while time.time() < deadline and not stop["flag"]:
            time.sleep(min(0.5, max(0.0, deadline - time.time())))
    fh.close()

    print("\n=== watchdog summary ===")
    print(f"elapsed_s={time.time() - t0:.0f} polls={stats['polls']} failures={stats['failures']} "
          f"alerts={stats['alerts']} (crash={stats['crash_alerts']}, hang={stats['hang_alerts']})")
    for n, v in stats["max_token_usage"].items():
        print(f"max_token_usage[{n}]={v:.4f}")
    for n, v in stats["peak_rss_kb"].items():
        print(f"peak_rss[{n}]={v / 1024:.1f} MiB")
    for t in logs.values():
        for sig, line in t.crash_hits[:20]:
            print(f"crash-sig {t.name} {sig!r}: {line[:200]}")
    print(f"csv={args.out}")
    sys.exit(2 if (stats["crash_alerts"] or stats["hang_alerts"]) else 0)


if __name__ == "__main__":
    main()
