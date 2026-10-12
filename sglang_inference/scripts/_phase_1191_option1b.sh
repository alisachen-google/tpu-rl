#!/bin/bash
# #1191 (validation-only head): (c) limit 0 + priority refused at start-up; (d) main with the same recipe: probe crashes all ranks.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; OUT=$R/fix_validate_1191_option1; mkdir -p $OUT; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
URL=http://10.128.0.90:30000; BR=pd-queue-limit-prefill; REC=$HOME/pd_disagg/recipes/gpt-oss_G_128k_fp8_e4m3_q0.json
log(){ echo "$(date -u +%FT%TZ) [1191o1b] $*"; }
while pgrep -f '_phase_1191_option''1.sh' >/dev/null; do sleep 30; done
cd $HOME/sglang-torchtpu && git fetch -q origin $BR && git checkout -q --detach origin/$BR && log "=== (c) on $(git log -1 --format=%h): monolithic limit 0 + priority scheduling, expect start refused"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
timeout 300 bash $S/launch_role.sh $REC golden o1191_refuse_q0ps > $OUT/refuse_q0ps.out 2>&1; echo "   launcher exit $?; message: $(grep -rhoE 'max-queued-requests 0 cannot be combined[^"]{0,160}' $OUT/refuse_q0ps.out $R/runs/o1191_refuse_q0ps/ 2>/dev/null | head -n1)"; echo "   server processes left: $(pgrep -fc 'sglang::scheduler')"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
git checkout -q main && log "=== (d) main $(git log -1 --format=%h): same recipe, probe q0 -> expect the scheduler to die"
env -u TORCH_DEVICE_BACKEND_AUTOLOAD timeout 3600 bash $S/pd.sh golden $REC G_main_q0ps b 2>&1 | grep -E 'ready|NOT'
if curl -sf -m 30 $URL/health >/dev/null; then
  $PY $S/queue_limit_probe.py --url $URL --decode $URL --mode q0 --slots 4 > $OUT/q0_mono_main.txt 2>&1; sed 's/^/   /' $OUT/q0_mono_main.txt | cut -c1-240
  sleep 5; echo "   health after: $(curl -s -m 5 -o /dev/null -w '%{http_code}' $URL/health); 'max() iterable argument is empty' in logs: $(grep -rhc 'max() iterable argument is empty' $R/runs/G_main_q0ps/ 2>/dev/null | awk '{s+=$1} END{print s+0}'); schedulers alive: $(pgrep -fc 'sglang::scheduler')"
else echo "   server not ready"; fi
bash $S/pd.sh stop >/dev/null 2>&1; log "phase 1191o1b done"
