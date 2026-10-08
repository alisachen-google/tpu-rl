#!/usr/bin/env bash
# F-12 repro: start a fresh gpt-oss Config A pair with max_running_requests 1 (same recipe as the stalled F11 128k_mr1 pair), send one 8k
# prompt with a 300 s timeout, record whether the P/D handshake completes. Repeat with a second fresh pair. Dump stacks if it stalls.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; A=http://10.128.15.217:30000; B=http://10.128.0.90:30000; OUT=$R/F12; mkdir -p $OUT
log(){ echo "$(date -u +%FT%TZ) $*"; }
for attempt in 1 2; do
  bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; bash $S/tpu_free.sh || { log "TPU not free"; exit 1; }
  RUN=F12_gptoss_A_128k_mr1_try$attempt; log "F-12 repro attempt $attempt: fresh pair $RUN"
  env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_A_128k_mr1.json $RUN 2>&1 | tee $R/runs/$RUN.driver.log | grep -E 'ready|NOT'
  for L in 8192 8192 32768; do
    t0=$(date +%s); $PY $S/golden_diff.py generate --prompts $HOME/pd_disagg/prompts/gpt-oss/$L.jsonl --url $LB --out $OUT/${RUN}_${L}_$t0.jsonl --max-new-tokens 64 --ignore-eos --concurrency 1 --timeout 300 > $OUT/${RUN}_${L}_$t0.log 2>&1 &
    cpid=$!; sleep 45
    if kill -0 $cpid 2>/dev/null; then
      log "   ISL=$L: no answer after 45 s -> stall suspected; capturing state"
      curl -s -m 5 $B/metrics | grep -E '^sglang:(num_running_reqs|num_decode_prealloc_queue_reqs|num_decode_transfer_queue_reqs)' | sed -E 's/\{[^}]*\}//' | tr '\n' ' '; echo
      D=$(pgrep -f 'sglang::scheduler' | head -1); sudo -n $HOME/venv/sglang/bin/py-spy dump --pid $D --nonblocking > $OUT/${RUN}_${L}_decode_pyspy.txt 2>&1 || $HOME/venv/sglang/bin/py-spy dump --pid $D --nonblocking > $OUT/${RUN}_${L}_decode_pyspy.txt 2>&1
      grep -E 'handshake|pop_preallocated|Decode batch|recv_requests' $OUT/${RUN}_${L}_decode_pyspy.txt | head -3
      wait $cpid
    fi
    echo "   ISL=$L: $(grep -cE 'tokens=64' $OUT/${RUN}_${L}_$t0.log) of 2 completed in $(( $(date +%s) - t0 )) s; $(grep -oE 'ERROR [A-Za-z]+' $OUT/${RUN}_${L}_$t0.log | sort | uniq -c | tr '\n' ' ')"
  done
  grep -hE 'handshake failed|transfer failed|Bootstrapping' $R/runs/$RUN/decode/server.log | head -n 2 | cut -c1-160
done
bash $S/pd.sh stop >/dev/null 2>&1; log "phase F12 repro done"
