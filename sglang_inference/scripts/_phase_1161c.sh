#!/usr/bin/env bash
# Preemption-on configurations for #1161 (C1/C2 threshold boundary at T=10 and T=0). Waits for 1161b.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; OUT=$R/1161; mkdir -p $OUT
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
log(){ echo "$(date -u +%FT%TZ) [1161c] $*"; }
while pgrep -f '[_]phase_1161b.sh|[_]phase_1161.sh|[_]phase_fix_scenarios.sh' >/dev/null; do sleep 30; done
cd $HOME/sglang-torchtpu && git checkout -q main
rsync -a -e "ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no" --rsync-path='sudo rsync' $HOME/pd_disagg/recipes/ sa_101411256518727260534@10.128.15.217:/home/alisachen_google_com/pd_disagg/recipes/
for cfg in ps_pre:10:gpt-oss_B_128k_ps_pre ps_pre_t0:0:gpt-oss_B_128k_ps_pre_t0; do
  name=${cfg%%:*}; rest=${cfg#*:}; thr=${rest%%:*}; rec=${rest#*:}
  log "=== $name ($rec, preemption on, threshold $thr)"
  bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
  env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/$rec.json 1161_$name 2>&1 | tee $R/runs/1161_$name.driver.log | grep -E 'ready|NOT'
  if curl -sf -m 30 $LB/health_generate >/dev/null; then
    $PY $S/priority_probe.py --url $LB --decode $DEC --mode pre --threshold $thr --slots 8 --hold-tokens 3000 > $OUT/$name.txt 2>&1; sed 's/^/   /' $OUT/$name.txt | cut -c1-230
    echo "   exceptions: prefill $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception' /home/alisachen_google_com/pd_disagg/results/runs/1161_$name/prefill/server.log") decode $(grep -cE 'Scheduler hit an exception' $R/runs/1161_$name/decode/server.log); preempt/retract log lines (decode): $(grep -iE 'preempt|retract' $R/runs/1161_$name/decode/server.log | grep -vcE 'server_args|retraction_backup|UserWarning')"
    echo "   health: prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"
  else log "$name: pair did not come up"; fi
done
bash $S/pd.sh stop >/dev/null 2>&1; log "phase 1161c done"
