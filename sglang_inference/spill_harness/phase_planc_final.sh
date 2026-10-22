#!/usr/bin/env bash
# Final validation for a run: the step-5 ladder (phase_planc_v2.sh) and a timed soak.
set -u
RUN=${RUN:?RUN}; SOAK_MIN=${SOAK_MIN:-15}; export LADDER=${LADDER:-"16 20 12"}
cd "$(dirname "$0")"
echo "$(date -u '+%F %T') FINAL start run=$RUN ladder=$LADDER soak=${SOAK_MIN}m"
bash phase_planc_v2.sh
echo "$(date -u '+%F %T') soak ${SOAK_MIN} min"
source ~/venv/sglang/bin/activate
python spill_stress_planc.py --phases soak --soak-min "$SOAK_MIN" --out "results/${RUN}.soak.json" 2>&1 | grep -E '"phase"|health|error' | cut -c1-400
bash peer_stats.sh "$RUN"
for u in http://10.128.15.217:30000 http://10.128.0.90:30000; do printf '%s %s\n' $u "$(curl -s -m 5 -o /dev/null -w '%{http_code}' $u/health)"; done
echo "$(date -u '+%F %T') FINAL DONE"
