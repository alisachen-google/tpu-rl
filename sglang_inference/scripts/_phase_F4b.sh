#!/usr/bin/env bash
# Replacement for the tail of _phase_goldens_1m_and_F4.sh (goldens are done): (ii) gpt-oss Config A 1m TC-1, (iii) F-4 isolation with
# --disable-hybrid-swa-memory. Emits the same end marker so _phase_A_gptoss_redo.sh continues.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; A=http://10.128.15.217:30000; B=http://10.128.0.90:30000
log(){ echo "$(date -u +%FT%TZ) $*"; }
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; bash $S/tpu_free.sh || { log "TPU not free, abort"; exit 1; }
RUN=A_gptoss_1m_run2; log "PD Config A gpt-oss 1m (host pool 320 GiB) -> $RUN"
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_A_1m.json $RUN 2>&1 | tee $R/runs/$RUN.driver.log
if curl -sf -m 30 $LB/health_generate >/dev/null; then
  nohup $PY $S/pd_watchdog.py --server prefill=$A --server decode=$B --log decode=$R/runs/$RUN/decode/server.log --interval 10 --hang-secs 600 --out $R/runs/$RUN/watchdog.csv > $R/runs/$RUN/watchdog.out 2>&1 &
  mkdir -p $R/A1m/tc1/gpt-oss
  bash $S/run_tc1.sh gpt-oss $LB $R/A1m/tc1/gpt-oss 262144,255123,524288,511777,1047552,999999 $R/golden/gpt-oss_fp8_1m_c1 1
  for L in 262144 255123 524288 511777 1047552 999999; do python3 $S/needle_check.py --prompts $HOME/pd_disagg/prompts/gpt-oss/$L.jsonl --out $R/A1m/tc1/gpt-oss/$L.jsonl --other $R/golden/gpt-oss_fp8_1m_c1/$L.jsonl --no-fail --report $R/A1m/tc1/gpt-oss/needles_$L.json 2>/dev/null | cut -c1-220; done
  log "boundary: ISL = context-1024 accepted, context+1 rejected"
  $PY $S/verify_pd_disagg.py --url $LB --router --lengths 1024 --boundary --context-length 1056768 --timeout 3600 2>&1 | tail -n 8 | tee $R/A1m/tc1/gpt-oss/boundary.txt
  curl -s -m 10 $B/metrics | grep -E '^sglang:kv_transfer_(latency_ms|total_mb|bootstrap_ms)_(sum|count)' | sed -E 's/\{[^}]*\}//' | sort -u | tr '\n' ' ' | tee $R/A1m/transfer_metrics.txt; echo
  bash $S/compile_audit.sh $R/runs/$RUN/decode/server.log decode_A1m | tee $R/A1m/compile_audit_decode.txt
else
  log "1m PD did not come up; see $R/runs/$RUN"; grep -hE 'ValueError|RuntimeError|Error:' $R/runs/$RUN/decode/server.log 2>/dev/null | tail -n 3 | cut -c1-200
fi
bash $S/pd.sh stop; sleep 5
RUN=A_gptoss_128k_noswamem; log "F-4 isolation: PD Config A gpt-oss 128k --disable-hybrid-swa-memory -> $RUN"
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_A_128k_noswamem.json $RUN 2>&1 | tee $R/runs/$RUN.driver.log
if curl -sf -m 30 $LB/health_generate >/dev/null; then
  mkdir -p $R/A_noswamem
  bash $S/run_tc1.sh gpt-oss $LB $R/A_noswamem/tc1/gpt-oss 8192,32768,131072 $R/golden/gpt-oss_fp8_rep2_c1 1
  echo "   chunk histogram (prefill, from decode-visible log is not available; see host a prefill log)"; grep -oE 'Prefill batch, #new-seq: 1, #new-token: [0-9]+' $R/runs/$RUN/decode/server.log | grep -oE '[0-9]+$' | sort -n | uniq -c | sort -rn | head -5
else
  log "noswamem PD did not come up"; grep -hE 'ValueError|RuntimeError|Error:' $R/runs/$RUN/decode/server.log 2>/dev/null | tail -n 3 | cut -c1-200
fi
log "phase goldens-1m + F4 done"
