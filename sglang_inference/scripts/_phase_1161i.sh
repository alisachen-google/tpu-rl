#!/usr/bin/env bash
# Cold vs cached output identity on Config B (with and without priority scheduling). Waits for 1161h.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; OUT=$R/1161; P=$HOME/pd_disagg/prompts/gpt-oss
log(){ echo "$(date -u +%FT%TZ) [1161i] $*"; }
while pgrep -f '[_]phase_1161h.sh' >/dev/null; do sleep 30; done
cd $HOME/sglang-torchtpu && git checkout -q main
for cfg in "gpt-oss_B_128k_ps_np:--priority" "gpt-oss_B_128k:"; do rec=${cfg%%:*}; flag=${cfg#*:}
  log "=== $rec"; bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
  env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/$rec.json 1161i_$rec 2>&1 | grep -E 'ready|NOT'
  curl -sf -m 30 $LB/health_generate >/dev/null && $PY $S/prefix_identity_probe.py --url $LB --prompts $P --n 256 $flag 2>&1 | sed 's/^/   /' | cut -c1-240
done
bash $S/pd.sh stop >/dev/null 2>&1; log "phase 1161i done"
