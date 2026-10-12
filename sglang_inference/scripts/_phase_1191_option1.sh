#!/bin/bash
# #1191 option 1: (a) monolithic limit 0 + priority: newcomers refused with 503, no crash; (b) PD + --max-queued-requests refuses to start.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/fix_validate_1191_option1; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
URL=http://10.128.0.90:30000; BR=pd-queue-limit-prefill
log(){ echo "$(date -u +%FT%TZ) [1191o1] $*"; }
cd $HOME/sglang-torchtpu && git fetch -q origin $BR && git checkout -q --detach origin/$BR && log "=== on $(git log -1 --format=%h)"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
log "(b) PD decode role with --max-queued-requests 0: expect start refused"
timeout 300 bash $S/launch_role.sh $HOME/pd_disagg/recipes/gpt-oss_C_128k_ps_q0.json decode o1191_refuse > $OUT/refuse_launch.out 2>&1; echo "   launcher exit $?; message: $(grep -rhoE 'max-queued-requests is not supported[^"]{0,140}' $OUT/refuse_launch.out $R/runs/o1191_refuse/ 2>/dev/null | head -n1)"; echo "   server processes left: $(pgrep -fc 'sglang::scheduler')"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
log "(a) monolithic limit 0 + priority scheduling: recipe gpt-oss_G_128k_fp8_e4m3_q0.json"
env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh golden $HOME/pd_disagg/recipes/gpt-oss_G_128k_fp8_e4m3_q0.json G_o1191_q0 b 2>&1 | grep -E 'ready|NOT'
if curl -sf -m 30 $URL/health >/dev/null; then
  $PY $S/queue_limit_probe.py --url $URL --decode $URL --mode q0 --slots 4 > $OUT/q0_mono.txt 2>&1; sed 's/^/   /' $OUT/q0_mono.txt | cut -c1-240
  echo "   health after: $(curl -s -m 5 -o /dev/null -w '%{http_code}' $URL/health); scheduler tracebacks: $(grep -rhcE 'Scheduler hit an exception|AssertionError|ValueError' $R/runs/G_o1191_q0/ 2>/dev/null | awk '{s+=$1} END{print s+0}')"
else echo "   server not ready"; fi
bash $S/pd.sh stop >/dev/null 2>&1; cd $HOME/sglang-torchtpu && git checkout -q main; log "phase 1191o1 done"
