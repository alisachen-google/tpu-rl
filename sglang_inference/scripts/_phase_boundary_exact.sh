#!/usr/bin/env bash
# Context-length boundary check with exact token counts (the heuristic run was a false alarm): Config A pair, 128k shape.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; SNAP=$(ls -d /mnt/data/tt/hf-cache/hub/models--openai--gpt-oss-120b/snapshots/* | head -1); OUT=$R/A/boundary; mkdir -p $OUT
log(){ echo "$(date -u +%FT%TZ) [boundary] $*"; }
until grep -q 'phase mtbench EC done' $R/phase_mtbench_EC.log 2>/dev/null; do sleep 120; done; log "mtbench done"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
RUN=A_gpt-oss_128k_boundary; env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_A_128k.json $RUN 2>&1 | grep -E 'ready|NOT'
if curl -sf -m 30 $LB/health_generate >/dev/null; then
  $PY $S/verify_pd_disagg.py --url $LB --router --lengths 1024 --boundary --context-length 140288 --tokenizer-json $SNAP/tokenizer.json --timeout 1800 2>&1 | grep -E 'tokenizer|boundary|summary|PASS|FAIL' | cut -c1-220 | tee $OUT/boundary_exact_128k.txt
else log "pair did not come up"; fi
bash $S/pd.sh stop >/dev/null 2>&1; log "phase boundary exact done"
