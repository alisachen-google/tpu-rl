#!/usr/bin/env bash
# #1161 second battery: np2 + soak (gpt-oss B PS-np), c2 (gpt-oss C PS-np), mm (Qwen B PS-np), pre2 (monolithic preemption).
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; OUT=$R/1161; mkdir -p $OUT
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
log(){ echo "$(date -u +%FT%TZ) [1161h2] $*"; }
excs(){ echo "   exceptions: prefill $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception' /home/alisachen_google_com/pd_disagg/results/runs/$1/prefill/server.log") decode $(grep -cE 'Scheduler hit an exception' $R/runs/$1/decode/server.log); health: prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"; }
start(){ bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done; env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $1 $2 2>&1 | tee $R/runs/$2.driver.log | grep -E 'ready|NOT'; curl -sf -m 30 $LB/health_generate >/dev/null; }
cd $HOME/sglang-torchtpu && git checkout -q main
rsync -a -e "ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no" --rsync-path='sudo rsync' $HOME/pd_disagg/recipes/ sa_101411256518727260534@10.128.15.217:/home/alisachen_google_com/pd_disagg/recipes/

log "=== c2: gpt-oss Config C (HiCache), priority scheduling, no preemption"
if start $HOME/pd_disagg/recipes/gpt-oss_C_128k_ps_np.json 1161h_C_np; then
  cd $S && $PY priority_probe2.py --url $LB --decode $DEC --mode c2 --slots 8 > $OUT/h_c2.txt 2>&1; sed 's/^/   /' $OUT/h_c2.txt | cut -c1-260
  echo "   prefill host-hit lines: $($SSH_A "sudo -n grep -E '#host-hit-token: [1-9]' /home/alisachen_google_com/pd_disagg/results/runs/1161h_C_np/prefill/server.log | tail -n 1 | grep -oE '#hbm-hit-token: [0-9]+, #host-hit-token: [0-9]+'")"; excs 1161h_C_np
else log "C pair did not come up"; fi

log "=== mm: Qwen3-VL Config B, priority scheduling, no preemption (E3 mixed priorities, E5 image prefix reuse)"
if start $HOME/pd_disagg/recipes/qwen3-vl_B_128k_ps_np.json 1161h_qwen_B_np; then
  for spec in "0 4 448x448" "10 4 448x448" "50 4 448x448"; do set -- $spec; ( $PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images $2 --resolution $3 --max-tokens 48 --temperature 0 --needle-word plover --priority $1 > $OUT/h_mm_p$1.log 2>&1 & ); done; sleep 1; wait; sleep 60
  for p in 0 10 50; do echo "   E3 priority $p: $(grep -E 'status=|needle' $OUT/h_mm_p$p.log | tail -n 2 | tr '\n' ' ' | cut -c1-200)"; done
  $PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images 4 --resolution 448x448 --max-tokens 48 --temperature 0 --needle-word plover --priority 0 --question "Describe each image briefly and then state the secret word." > $OUT/h_mm_e5a.log 2>&1
  $PY $S/mm_request.py --url $LB --model qwen3-vl-30b --num-images 4 --resolution 448x448 --max-tokens 48 --temperature 0 --needle-word plover --priority 50 --question "Describe each image briefly and then state the secret word." > $OUT/h_mm_e5b.log 2>&1
  echo "   E5 same images, priority 0 then 50: $(grep -oE 'cached_tokens=[0-9A-Za-z]+' $OUT/h_mm_e5a.log | tail -n 1) -> $(grep -oE 'cached_tokens=[0-9A-Za-z]+' $OUT/h_mm_e5b.log | tail -n 1); needles: $(grep -cE 'needle FOUND|found' $OUT/h_mm_e5a.log $OUT/h_mm_e5b.log | tr '\n' ' ')"
  excs 1161h_qwen_B_np
else log "Qwen pair did not come up"; fi

log "=== pre2: monolithic gpt-oss, preemption on (C9 C10 C15)"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh golden $HOME/pd_disagg/recipes/gpt-oss_G_128k_ps_pre.json 1161h_mono_pre b 2>&1 | grep -E 'ready|NOT'
if curl -sf -m 30 $DEC/health >/dev/null; then
  cd $S && $PY priority_probe2.py --url $DEC --decode $DEC --mode pre2 --slots 4 > $OUT/h_pre2.txt 2>&1; sed 's/^/   /' $OUT/h_pre2.txt | cut -c1-260
  L=$(ls $R/runs/1161h_mono_pre/*/server.log 2>/dev/null | head -n 1); echo "   exceptions: $(grep -cE 'Scheduler hit an exception' $L); health $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"
else log "mono server did not come up"; fi
bash $S/pd.sh stop >/dev/null 2>&1; log "phase 1161h2 done"
