#!/usr/bin/env bash
# #1161 priority scheduling, first battery (gpt-oss, 128k): PS-np, PS-pre, PS-low, PS-q, PS-dis. Queued behind the F-8 upstream check.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; PRE=http://10.128.15.217:30000; DEC=http://10.128.0.90:30000; OUT=$R/1161; mkdir -p $OUT
SSH_A="ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no sa_101411256518727260534@10.128.15.217"
log(){ echo "$(date -u +%FT%TZ) [1161] $*"; }
while pgrep -f '[_]phase_f8_upstream_check.sh' >/dev/null; do sleep 30; done
cd $HOME/sglang-torchtpu && git checkout -q main
rsync -a -e "ssh -i $HOME/.ssh/google_compute_engine -o BatchMode=yes -o StrictHostKeyChecking=no" --rsync-path='sudo rsync' $HOME/pd_disagg/recipes/ sa_101411256518727260534@10.128.15.217:/home/alisachen_google_com/pd_disagg/recipes/
start(){ bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done; env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $1 $2 2>&1 | tee $R/runs/$2.driver.log | grep -E 'ready|NOT'; curl -sf -m 30 $LB/health_generate >/dev/null; }
excs(){ echo "   exceptions: prefill $($SSH_A "sudo -n grep -cE 'Scheduler hit an exception' /home/alisachen_google_com/pd_disagg/results/runs/$1/prefill/server.log") decode $(grep -cE 'Scheduler hit an exception' $R/runs/$1/decode/server.log); preempt/retract log lines (decode, excl. args): $(grep -iE 'preempt|retract' $R/runs/$1/decode/server.log | grep -vcE 'server_args|retraction_backup|UserWarning')"; }
for cfg in ps_np:np:gpt-oss_B_128k_ps_np ps_low:low:gpt-oss_B_128k_ps_low ps_q:q:gpt-oss_C_128k_ps_q ps_dis:dis:gpt-oss_B_128k_ps_dis; do
  name=${cfg%%:*}; rest=${cfg#*:}; mode=${rest%%:*}; rec=${rest#*:}
  log "=== $name ($rec, mode $mode)"
  if start $HOME/pd_disagg/recipes/$rec.json 1161_$name; then
    $PY $S/priority_probe.py --url $LB --decode $DEC --mode $mode --slots 8 --hold-tokens 3000 > $OUT/$name.txt 2>&1; sed 's/^/   /' $OUT/$name.txt | cut -c1-230
    excs 1161_$name; echo "   health: prefill $(curl -s -m 5 -o /dev/null -w '%{http_code}' $PRE/health) decode $(curl -s -m 5 -o /dev/null -w '%{http_code}' $DEC/health)"
  else log "$name: pair did not come up"; $SSH_A "sudo -n grep -E 'Error|error' /home/alisachen_google_com/pd_disagg/results/runs/1161_$name/prefill/server.log | grep -vE UserWarning | tail -n 2 | cut -c1-200" | sed 's/^/   prefill: /'; fi
done
bash $S/pd.sh stop >/dev/null 2>&1; log "phase 1161 done"
