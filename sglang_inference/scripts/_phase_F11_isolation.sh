#!/usr/bin/env bash
# F-11 isolation, after the F-2 patch test: (a) 1M shape with max_running_requests 4; (b) 128k shape with max_running_requests 1.
# For each: start, read token_capacity_swa, send one 32k prompt with a 900 s timeout, record progress or hang, abort, stop.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; A=http://10.128.15.217:30000; OUT=$R/F11; mkdir -p $OUT
log(){ echo "$(date -u +%FT%TZ) $*"; }
until grep -q 'phase F2 patch test done' $R/phase_F2_patch_test.log 2>/dev/null; do sleep 60; done; log "F2 patch test done"
for rec in gpt-oss_A_1m_mr4 gpt-oss_A_128k_mr1; do
  bash $S/pd.sh stop >/dev/null 2>&1; sleep 5; bash $S/tpu_free.sh || { log "TPU not free"; exit 1; }
  log "F-11 isolation: $rec"
  env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/$rec.json F11_$rec 2>&1 | tee $R/runs/F11_$rec.driver.log | grep -E 'ready|NOT'
  curl -sf -m 30 $LB/health_generate >/dev/null || { log "$rec did not come up"; continue; }
  curl -s -m 10 $A/get_server_info > $OUT/${rec}_prefill_server_info.json
  python3 -c "import json; d=json.load(open('$OUT/${rec}_prefill_server_info.json')); st=d['internal_states'][0]; print('   ', '$rec', 'mrr=',d.get('max_running_requests'), 'ctx=',d.get('context_length'), 'capacity=',st['memory_usage'].get('token_capacity'), 'capacity_swa=',st['memory_usage'].get('token_capacity_swa'))"
  for L in 8192 32768 131072; do
    t0=$(date +%s); $PY $S/golden_diff.py generate --prompts $HOME/pd_disagg/prompts/gpt-oss/$L.jsonl --url $LB --out $OUT/${rec}_$L.jsonl --max-new-tokens 64 --ignore-eos --concurrency 1 --timeout 900 > $OUT/${rec}_$L.log 2>&1
    echo "   ISL=$L: $(grep -cE 'tokens=64' $OUT/${rec}_$L.log) of 2 completed in $(( $(date +%s) - t0 )) s; swa_usage now: $(curl -s -m 10 $A/metrics | grep -E '^sglang:swa_token_usage' | sed -E 's/\{[^}]*\}//' | head -1)"
    sleep 20
  done
done
bash $S/pd.sh stop >/dev/null 2>&1
log "phase F11 isolation done"
