#!/usr/bin/env bash
# Re-run of the no-preemption configuration with the corrected (staggered) ordering probe; waits for the scenario battery.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; OUT=$R/1161; mkdir -p $OUT
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
log(){ echo "$(date -u +%FT%TZ) [1161f] $*"; }
while pgrep -f '[_]phase_1161e.sh|[_]phase_1161d.sh|[_]phase_1161c.sh' >/dev/null; do sleep 30; done
cd $HOME/sglang-torchtpu && git checkout -q main
log "=== ps_pre rerun (refilled slots for C2)"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_B_128k_ps_pre.json 1161_ps_pre_b 2>&1 | tee $R/runs/1161_ps_pre_b.driver.log | grep -E 'ready|NOT'
if curl -sf -m 30 $LB/health_generate >/dev/null; then
  $PY $S/priority_probe.py --url $LB --decode $DEC --mode pre --threshold 10 --slots 8 --hold-tokens 3000 > $OUT/ps_pre_b.txt 2>&1; sed 's/^/   /' $OUT/ps_pre_b.txt | cut -c1-230
  echo "   exceptions: prefill $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception' /home/alisachen_google_com/pd_disagg/results/runs/1161_ps_pre_b/prefill/server.log") decode $(grep -cE 'Scheduler hit an exception' $R/runs/1161_ps_pre_b/decode/server.log)"
else log "pair did not come up"; fi
bash $S/pd.sh stop >/dev/null 2>&1; log "phase 1161f done"
