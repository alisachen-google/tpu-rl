#!/usr/bin/env bash
# Wait for golden work to finish on both hosts, then switch to PD Config A (gpt-oss, 128k shape) and run TC-0/TC-1.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
log(){ echo "$(date -u +%FT%TZ) $*"; }
until grep -q 'run_golden finished' $R/golden/qwen3-vl_fp8/run.log 2>/dev/null; do sleep 30; done; log "qwen golden done"
until grep -q '=== .* done' $R/golden/gpt-oss_fp8/gsm8k.log 2>/dev/null; do sleep 30; done; log "gpt-oss extra (determinism+gsm8k) done"
# snapshot compile audits of the goldens before stopping
bash $S/compile_audit.sh $R/runs/G_qwen_128k_fp8/golden/server.log qwen_golden_b > $R/runs/G_qwen_128k_fp8/compile_audit.txt
bash $S/pd.sh stop; sleep 5; bash $S/tpu_free.sh || { log "TPU not free, abort"; exit 1; }
RUN=A_gptoss_128k_run1
log "starting PD Config A gpt-oss 128k -> $RUN"
bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_A_128k.json $RUN 2>&1 | tee $R/runs/$RUN.driver.log
# watchdog on both engines (prefill log is on host a; poll via /get_server_info only)
nohup $PY $S/pd_watchdog.py --server prefill=http://10.128.15.217:30000 --server decode=http://10.128.0.90:30000 --log decode=$R/runs/$RUN/decode/server.log --interval 10 --hang-secs 300 --out $R/runs/$RUN/watchdog.csv > $R/runs/$RUN/watchdog.out 2>&1 &
log "TC-1 gpt-oss via LB"
bash $S/run_tc1.sh gpt-oss http://10.128.15.217:8080 $R/A/tc1/gpt-oss 1024,8192,32768,131072,127999 $R/golden/gpt-oss_fp8 1
bash $S/pd.sh info $RUN
log "phase A gpt-oss TC-1 done"
