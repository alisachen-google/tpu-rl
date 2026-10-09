#!/usr/bin/env bash
# Monolithic control for preemption (C1/C2): preempt_to_schedule runs only in the prefill-admission path, so test it where it applies.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
URL=http://10.128.0.90:30000; OUT=$R/1161; mkdir -p $OUT
log(){ echo "$(date -u +%FT%TZ) [1161g] $*"; }
while pgrep -f '[_]phase_1161d.sh|[_]phase_1161e.sh' >/dev/null; do sleep 30; done
cd $HOME/sglang-torchtpu && git checkout -q main
log "=== mono ps_pre (monolithic gpt-oss 128k, 4 slots, preemption on, threshold 10)"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh golden $HOME/pd_disagg/recipes/gpt-oss_G_128k_ps_pre.json 1161_mono_ps_pre b 2>&1 | grep -E 'ready|NOT'
if curl -sf -m 30 $URL/health >/dev/null; then
  $PY $S/priority_probe.py --url $URL --decode $URL --mode pre --threshold 10 --slots 4 --hold-tokens 3000 > $OUT/mono_ps_pre.txt 2>&1; sed 's/^/   /' $OUT/mono_ps_pre.txt | cut -c1-230
  L=$(ls $R/runs/1161_mono_ps_pre/*/server.log 2>/dev/null | head -n 1); echo "   exceptions: $(grep -cE 'Scheduler hit an exception' $L); preempt/retract log lines: $(grep -iE 'preempt|retract' $L | grep -vcE 'server_args|retraction_backup|UserWarning')"
else log "mono server did not come up"; fi
bash $S/pd.sh stop >/dev/null 2>&1; log "phase 1161g done"
