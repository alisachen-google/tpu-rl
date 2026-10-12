#!/bin/bash
# TP-4 1P/1D baseline on the pair (p1 on a chips 0,1; d1 on b chips 0,1): router on b:8081, smoke, routing probe, 8k differential vs TP-8 golden, decode-phase abort.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/2p2d/tp4_1p1d; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
A=10.128.15.217; B=10.128.0.90; LB=http://$B:8081
log(){ echo "$(date -u +%FT%TZ) [tp4-1p1d] $*"; }
source $S/env_singlehost_v7x.sh >/dev/null 2>&1
pkill -u $(id -un) -f 'sglang_router.launch_router.*--port 8081' 2>/dev/null; sleep 1
setsid nohup $PYTHON -m sglang_router.launch_router --pd-disaggregation --mini-lb --prefill http://$A:30000 8998 --decode http://$B:30000 --host 0.0.0.0 --port 8081 > $OUT/router.log 2>&1 < /dev/null &
for i in $(seq 1 40); do curl -sf -m 5 $LB/health >/dev/null 2>&1 && { log "router up after $((i*3)) s"; break; }; sleep 3; done
log "router health $(curl -s -m 5 -o /dev/null -w '%{http_code}' $LB/health); engines: p1 $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$A:30000/health) d1 $(curl -s -m 5 -o /dev/null -w '%{http_code}' http://$B:30000/health)"
log "smoke: $(curl -s -m 300 $LB/generate -H 'Content-Type: application/json' -d '{"text":"The three primary colors are","sampling_params":{"temperature":0,"max_new_tokens":16}}' | head -c 300)"
log "=== routing probe (1P/1D degenerate: checks health, serving, pair-invariant output, idle)"; $PY $S/route_probe.py --router $LB --prefills http://$A:30000 --decodes http://$B:30000 --n 12 --concurrency 4 2>&1 | tee $OUT/route_probe.txt | sed 's/^/   /' | cut -c1-220
log "=== TC-1 differential: 8k prompts, greedy 1024, c=1, vs TP-8 golden (expect NUM class at TP-4)"; $PY $S/golden_diff.py generate --prompts $HOME/pd_disagg/prompts/gpt-oss/8192.jsonl --url $LB --out $OUT/8192.jsonl --max-new-tokens 1024 --ignore-eos --return-logprob --concurrency 1 > $OUT/gen_8192.log 2>&1; $PY $S/golden_diff.py diff --a $R/golden/gpt-oss_fp8/8192.jsonl --b $OUT/8192.jsonl --report $OUT/diff_8192.json 2>&1 | tail -n 8 | sed 's/^/   /' | cut -c1-220
log "=== TC-7 decode-phase abort x2"; $PY $S/abort_test.py --url $LB --prompt-file $HOME/pd_disagg/prompts/gpt-oss/8192.jsonl --phase decode --abort-after 5 --repeat 2 --server-info http://$A:30000 --server-info http://$B:30000 --settle 60 > $OUT/abort_decode.log 2>&1; tail -n 5 $OUT/abort_decode.log | sed 's/^/   /' | cut -c1-220
log "=== compile audit d1: $(bash $S/compile_audit.sh $R/runs/2p2d_bringup2/d1/server.log 2>/dev/null | tail -n 1 | cut -c1-160)"
log "phase tp4-1p1d done (engines left running)"
