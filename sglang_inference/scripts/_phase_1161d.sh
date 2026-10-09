#!/usr/bin/env bash
# Monolithic control for the queue-limit cases (D1-D3): the limit is enforced only in non-PD mode, so verify it there. Waits for 1161c.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
URL=http://10.128.0.90:30000; OUT=$R/1161; mkdir -p $OUT
log(){ echo "$(date -u +%FT%TZ) [1161d] $*"; }
while pgrep -f '[_]phase_1161c.sh|[_]phase_1161b.sh|[_]phase_1161.sh|[_]phase_fix_scenarios.sh' >/dev/null; do sleep 30; done
cd $HOME/sglang-torchtpu && git checkout -q main
log "=== mono ps_q (monolithic gpt-oss 128k, 4 slots, max_queued_requests 4, priority scheduling, no preemption)"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh golden $HOME/pd_disagg/recipes/gpt-oss_G_128k_ps_q.json 1161_mono_ps_q b 2>&1 | grep -E 'ready|NOT'
if curl -sf -m 30 $URL/health >/dev/null; then
  $PY $S/priority_probe.py --url $URL --decode $URL --mode q --slots 4 --hold-tokens 3000 > $OUT/mono_ps_q.txt 2>&1; sed 's/^/   /' $OUT/mono_ps_q.txt | cut -c1-230
  L=$(ls $R/runs/1161_mono_ps_q/*/server.log 2>/dev/null | head -n 1); echo "   exceptions: $(grep -cE 'Scheduler hit an exception' $L); queue-full aborts logged: $(grep -ciE 'queue is full|higher priority request' $L)"
else log "mono server did not come up"; fi
bash $S/pd.sh stop >/dev/null 2>&1; log "phase 1161d done"
