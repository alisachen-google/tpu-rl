#!/usr/bin/env python3
"""Generate grafana/dashboards/sglang_pd.json from the panel list (Overall / P / D rows)."""
import json

def hist(metric, sel):
    b = f'rate(sglang:{metric}_bucket{sel}[5m])'
    return [
        (f'histogram_quantile(0.50, sum by (le) ({b}))', 'p50'),
        (f'histogram_quantile(0.95, sum by (le) ({b}))', 'p95'),
        (f'sum(rate(sglang:{metric}_sum{sel}[5m])) / sum(rate(sglang:{metric}_count{sel}[5m]))', 'mean'),
    ]

P = '{engine_type="prefill"}'; D = '{engine_type="decode"}'
def stage(role, st): return '{engine_type="%s",stage="%s"}' % (role, st)

ROWS = [
  ("Overall", [
    ("Total requests/s (decode side)", [(f'sum(rate(sglang:num_requests_total{D}[1m]))', 'req/s')], 'reqps', ""),
    ("Aborted requests/s", [(f'sum(rate(sglang:num_aborted_requests_total{D}[5m]))', 'decode'), (f'sum(rate(sglang:num_aborted_requests_total{P}[5m]))', 'prefill')], 'reqps', ""),
    ("TTFT (decode side)", hist('time_to_first_token_seconds', D), 's', ""),
    ("Inter-token latency (decode side)", hist('inter_token_latency_seconds', D), 's', "Not exported by this build (09d1244c26); panel is ready if the metric appears."),
    ("E2E request latency (decode side)", hist('e2e_request_latency_seconds', D), 's', ""),
    ("Output tokens/s (decode)", [(f'sum(rate(sglang:realtime_tokens_total{{engine_type="decode",mode="decode"}}[1m]))', 'realtime decode'), (f'sglang:gen_throughput{{engine_type="decode",tp_rank="0"}}', 'gen_throughput')], 'short', ""),
  ]),
  ("P (prefill)", [
    ("P bootstrap queue depth", [(f'sglang:num_prefill_bootstrap_queue_reqs{{engine_type="prefill",tp_rank="0"}}', 'bootstrap Q')], 'short', ""),
    ("P bootstrap duration", hist('per_stage_req_latency_seconds', stage('prefill', 'prefill_bootstrap')), 's', ""),
    ("P bootstrap failures/s", [(f'sum(rate(sglang:num_bootstrap_failed_reqs_total{P}[5m]))', 'failures/s')], 'short', ""),
    ("P ordinary queue depth", [(f'sglang:num_queue_reqs{{engine_type="prefill",tp_rank="0"}}', 'queue')], 'short', ""),
    ("P ordinary queue wait", hist('queue_time_seconds', P), 's', ""),
    ("P running batch size", [(f'sglang:num_running_reqs{{engine_type="prefill",tp_rank="0"}}', 'running')], 'short', ""),
    ("P prefill duration (forward)", hist('per_stage_req_latency_seconds', stage('prefill', 'prefill_forward')), 's', ""),
    ("P chunked prefill duration", hist('per_stage_req_latency_seconds', stage('prefill', 'chunked_prefill')), 's', ""),
    ("P final transfer queue depth (inflight)", [(f'sglang:num_prefill_inflight_queue_reqs{{engine_type="prefill",tp_rank="0"}}', 'inflight')], 'short', ""),
    ("P final transfer-wait duration", hist('per_stage_req_latency_seconds', stage('prefill', 'prefill_transfer_kv_cache')), 's', ""),
    ("P oldest transfer age", [(f'max(sglang:kv_transfer_oldest_inflight_age_seconds{P})', 'oldest inflight')], 's', ""),
    ("P transfer failures/s", [(f'sum(rate(sglang:num_transfer_failed_reqs_total{P}[5m]))', 'failures/s')], 'short', ""),
    ("P computed vs cached tokens/s", [(f'sum(rate(sglang:realtime_tokens_total{{engine_type="prefill",mode="prefill_compute"}}[1m]))', 'computed'), (f'sum(rate(sglang:realtime_tokens_total{{engine_type="prefill",mode="prefill_cache"}}[1m]))', 'cached')], 'short', ""),
    ("P KV usage (full / SWA)", [(f'sglang:token_usage{{engine_type="prefill",tp_rank="0"}}', 'full'), (f'sglang:swa_token_usage{{engine_type="prefill",tp_rank="0"}}', 'swa')], 'percentunit', "SWA usage pinned near 1.0 with a stalled chunk trail is the F-11 signature."),
  ]),
  ("D (decode)", [
    ("D bootstrap+prealloc queue depth", [(f'sglang:num_decode_prealloc_queue_reqs{{engine_type="decode",tp_rank="0"}}', 'prealloc Q')], 'short', "Held requests with no decode batches is the F-12 signature."),
    ("D bootstrap+prealloc duration", hist('per_stage_req_latency_seconds', stage('decode', 'decode_bootstrap')), 's', ""),
    ("D bootstrap failures/s", [(f'sum(rate(sglang:num_bootstrap_failed_reqs_total{D}[5m]))', 'failures/s')], 'short', ""),
    ("D KV transfer queue depth", [(f'sglang:num_decode_transfer_queue_reqs{{engine_type="decode",tp_rank="0"}}', 'transfer Q')], 'short', ""),
    ("D KV handoff-wait duration", hist('per_stage_req_latency_seconds', stage('decode', 'decode_transferred')), 's', ""),
    ("D transfer failures/s", [(f'sum(rate(sglang:num_transfer_failed_reqs_total{D}[5m]))', 'failures/s')], 'short', ""),
    ("D ordinary queue depth", [(f'sglang:num_queue_reqs{{engine_type="decode",tp_rank="0"}}', 'queue')], 'short', ""),
    ("D ordinary queue wait", hist('queue_time_seconds', D), 's', ""),
    ("D running batch size", [(f'sglang:num_running_reqs{{engine_type="decode",tp_rank="0"}}', 'running')], 'short', ""),
    ("D retracted requests (gauge)", [(f'sglang:num_retracted_reqs{{engine_type="decode",tp_rank="0"}}', 'retracted'), (f'sglang:num_paused_reqs{{engine_type="decode",tp_rank="0"}}', 'paused')], 'short', "This build exports a per-interval gauge, not a counter."),
    ("D output tokens/s", [(f'sum(rate(sglang:realtime_tokens_total{{engine_type="decode",mode="decode"}}[1m]))', 'tokens/s')], 'short', ""),
    ("D total context per batch", [(f'sglang:decode_sum_seq_lens{{engine_type="decode",tp_rank="0"}}', 'sum seq lens')], 'short', ""),
    ("D average context per running request", [(f'sglang:decode_sum_seq_lens{{engine_type="decode",tp_rank="0"}} / sglang:num_running_reqs{{engine_type="decode",tp_rank="0"}}', 'avg ctx')], 'short', ""),
    ("D KV usage", [(f'sglang:token_usage{{engine_type="decode",tp_rank="0"}}', 'full'), (f'sglang:swa_token_usage{{engine_type="decode",tp_rank="0"}}', 'swa')], 'percentunit', ""),
  ]),
]

panels, pid, y = [], 1, 0
for title, items in ROWS:
    panels.append({"type": "row", "title": title, "id": pid, "collapsed": False, "gridPos": {"h": 1, "w": 24, "x": 0, "y": y}, "panels": []}); pid += 1; y += 1
    for i, (ptitle, targets, unit, desc) in enumerate(items):
        x = (i % 3) * 8
        panels.append({
            "type": "timeseries", "title": ptitle, "id": pid, "description": desc,
            "gridPos": {"h": 7, "w": 8, "x": x, "y": y},
            "datasource": {"type": "prometheus", "uid": "prom"},
            "fieldConfig": {"defaults": {"unit": unit, "custom": {"lineWidth": 1, "fillOpacity": 8, "showPoints": "never"}}, "overrides": []},
            "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}},
            "targets": [{"refId": chr(65 + k), "expr": e, "legendFormat": lf, "datasource": {"type": "prometheus", "uid": "prom"}} for k, (e, lf) in enumerate(targets)],
        }); pid += 1
        if i % 3 == 2: y += 7
    if len(items) % 3: y += 7

dash = {"uid": "sglang-pd", "title": "SGLang PD pair", "tags": ["sglang", "pd"], "timezone": "utc", "schemaVersion": 39, "version": 1,
        "refresh": "5s", "time": {"from": "now-30m", "to": "now"}, "editable": True, "panels": panels,
        "templating": {"list": []}, "annotations": {"list": []}}
json.dump(dash, open("grafana/dashboards/sglang_pd.json", "w"), indent=1)
print(f"dashboard: {len(panels)} panels")
