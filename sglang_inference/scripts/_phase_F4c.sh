#!/usr/bin/env bash
# After stopping the hung 1m-shape run (F-11): run only the F-4 isolation (Config A gpt-oss 128k with --disable-hybrid-swa-memory),
# then emit the end marker so _phase_A_gptoss_redo.sh continues.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; A=http://10.128.15.217:30000; B=http://10.128.0.90:30000
log(){ echo "$(date -u +%FT%TZ) $*"; }
bash $S/pd.sh stop >/dev/null 2>&1; sleep 5; bash $S/tpu_free.sh || { log "TPU not free, abort"; exit 1; }
RUN=A_gptoss_128k_noswamem; log "F-4 isolation: PD Config A gpt-oss 128k --disable-hybrid-swa-memory -> $RUN"
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_A_128k_noswamem.json $RUN 2>&1 | tee $R/runs/$RUN.driver.log
if curl -sf -m 30 $LB/health_generate >/dev/null; then
  mkdir -p $R/A_noswamem; curl -s -m 10 $A/get_server_info > $R/A_noswamem/prefill_server_info.json
  python3 -c "import json; d=json.load(open('$R/A_noswamem/prefill_server_info.json')); st=d['internal_states'][0]; print('   noswamem prefill:', {k:d.get(k) for k in ['disable_hybrid_swa_memory','max_total_num_tokens']}, {k:st['memory_usage'].get(k) for k in ['token_capacity','token_capacity_swa']})"
  bash $S/run_tc1.sh gpt-oss $LB $R/A_noswamem/tc1/gpt-oss 8192,32768,131072 $R/golden/gpt-oss_fp8_rep2_c1 1
  log "chunk histogram on the prefill (host a):"; ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217 "sudo -n grep -oE 'Prefill batch, #new-seq: 1, #new-token: [0-9]+' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log | grep -oE '[0-9]+\$' | sort -n | uniq -c | sort -rn | head -5"
else
  log "noswamem PD did not come up"; grep -hE 'ValueError|RuntimeError|Error:' $R/runs/$RUN/decode/server.log 2>/dev/null | tail -n 3 | cut -c1-200
fi
log "phase goldens-1m + F4 done"
