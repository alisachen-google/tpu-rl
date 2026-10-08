#!/usr/bin/env bash
# F-11 control + F-12 isolation: 128k recipe with max_running_requests 2. Prediction from the F-11 analysis: SWA pool 17216 -> 16896 after
# rounding (512 headroom), so 32k and 131k prompts must complete. If the first 8k request stalls, F-12 does not need a single slot.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; A=http://10.128.15.217:30000; B=http://10.128.0.90:30000; OUT=$R/F11; mkdir -p $OUT
log(){ echo "$(date -u +%FT%TZ) $*"; }
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; bash $S/tpu_free.sh || { log "TPU not free"; exit 1; }
RUN=F11_gpt-oss_A_128k_mr2; log "F-11 control: $RUN"
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_A_128k_mr2.json $RUN 2>&1 | tee $R/runs/$RUN.driver.log | grep -E 'ready|NOT'
ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217 "sudo -n grep -m1 -oE 'aligning swa_max_total_num_tokens [0-9]+ -> [0-9]+' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log"
for L in 8192 32768 131072; do
  t0=$(date +%s); $PY $S/golden_diff.py generate --prompts $HOME/pd_disagg/prompts/gpt-oss/$L.jsonl --url $LB --out $OUT/${RUN}_$L.jsonl --max-new-tokens 64 --ignore-eos --concurrency 1 --timeout 300 > $OUT/${RUN}_$L.log 2>&1 &
  cpid=$!; sleep 45
  if kill -0 $cpid 2>/dev/null; then
    log "   ISL=$L: no answer after 45 s -> stall suspected"; curl -s -m 5 $A/metrics | grep -E '^sglang:(swa_token_usage|num_running_reqs|num_prefill_inflight_queue_reqs)' | sed -E 's/\{[^}]*\}//' | tr '\n' ' '; echo
    D=$(pgrep -f 'sglang::scheduler' | head -1); $HOME/venv/sglang/bin/py-spy dump --pid $D --nonblocking 2>/dev/null | grep -E 'handshake|pop_preallocated|recv_requests' | head -2
    ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217 "sudo -n grep -E 'Prefill batch' /home/alisachen_google_com/pd_disagg/results/runs/$RUN/prefill/server.log | grep -vE '#new-token: 1,' | tail -n 3 | grep -oE '#new-token: [0-9]+.*swa token usage: [0-9.]+'"
    wait $cpid
  fi
  echo "   ISL=$L: $(grep -cE 'tokens=64' $OUT/${RUN}_$L.log) of 2 completed in $(( $(date +%s) - t0 )) s; $(grep -oE 'ERROR [A-Za-z]+' $OUT/${RUN}_$L.log | sort | uniq -c | tr '\n' ' ')"
done
bash $S/pd.sh stop >/dev/null 2>&1; log "phase F11 control mr2 done"
