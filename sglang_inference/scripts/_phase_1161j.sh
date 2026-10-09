#!/usr/bin/env bash
# #1161 E4: Qwen3-VL at the 256k shape with priority scheduling (no preemption): a 255k low-priority text request and a high-priority
# 4-image request in flight together; both must complete, no exceptions, resources back to zero.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; OUT=$R/1161; Q=$HOME/pd_disagg/prompts/qwen3-vl
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
log(){ echo "$(date -u +%FT%TZ) [1161j] $*"; }
cd $HOME/sglang-torchtpu && git checkout -q main
rsync -a -e "ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no" --rsync-path='sudo rsync' $HOME/pd_disagg/recipes/ sa_101411256518727260534@10.128.15.217:/home/alisachen_google_com/pd_disagg/recipes/
log "=== E4: Qwen3-VL Config B 256k, priority scheduling, no preemption"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/qwen3-vl_B_256k_ps_np.json 1161j_qwen_B_256k 2>&1 | tee $R/runs/1161j_qwen_B_256k.driver.log | grep -E 'ready|NOT'
if curl -sf -m 30 $LB/health_generate >/dev/null; then
  t0=$(date +%s)
  ( $PY $S/golden_diff.py generate --prompts $Q/255123.jsonl --url $LB --out $OUT/j_low_255k.jsonl --max-new-tokens 256 --ignore-eos --concurrency 1 --timeout 1800 --stream > $OUT/j_low_255k.log 2>&1 ) &
  sleep 20
  $PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images 4 --resolution 448x448 --max-tokens 48 --temperature 0 --needle-word plover --priority 50 > $OUT/j_high_mm.log 2>&1
  echo "   high-priority 4-image request (sent 20 s after the 255k low-priority prompt): $(grep -E 'status=' $OUT/j_high_mm.log | tail -n 1 | cut -c1-160)"
  wait; echo "   low-priority 255k text: $(grep -E 'tokens=|ERROR' $OUT/j_low_255k.log | head -n 1 | cut -c1-140) (total $(( $(date +%s)-t0 ))s)"
  echo "   exceptions: prefill $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception' /home/alisachen_google_com/pd_disagg/results/runs/1161j_qwen_B_256k/prefill/server.log") decode $(grep -cE 'Scheduler hit an exception' $R/runs/1161j_qwen_B_256k/decode/server.log); used tokens $(curl -s -m 10 $DEC/metrics | grep -E '^sglang:num_used_tokens' | head -n 1 | sed -E 's/.* //'); health prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"
else log "pair did not come up"; $SSH_A "sudo -n grep -E 'Error|error' /home/alisachen_google_com/pd_disagg/results/runs/1161j_qwen_B_256k/prefill/server.log | grep -vE UserWarning | tail -n 2 | cut -c1-200"; fi
bash $S/pd.sh stop >/dev/null 2>&1; log "phase 1161j done"
