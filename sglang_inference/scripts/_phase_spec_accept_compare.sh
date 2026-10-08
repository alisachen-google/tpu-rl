#!/usr/bin/env bash
# Controlled EAGLE3 acceptance comparison: same 20 natural-text prompts, c=1, per-request completion_tokens/spec_verify_ct,
# through the PD pair (Config D) and through the monolithic EAGLE3 server (bf16). Also a 32k long-context prompt pair for both.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; U=http://10.128.0.90:30000; OUT=$R/D/gpt-oss/spec_accept_compare; mkdir -p $OUT
log(){ echo "$(date -u +%FT%TZ) [spec-accept] $*"; }
until grep -q 'phase kv checksum qwen done' $R/phase_kv_checksum_qwen.log 2>/dev/null; do sleep 120; done; log "kv checksum phases done"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
log "=== PD Config D"
RUN=D_gpt-oss_128k_specprobe; env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_D_128k.json $RUN 2>&1 | grep -E 'ready|NOT'
if curl -sf -m 30 $LB/health_generate >/dev/null; then $PY $S/spec_accept_probe.py --url $LB --n 20 --max-new-tokens 256 --out $OUT/pd_D.jsonl; else log "PD pair did not come up"; fi
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done
log "=== monolithic EAGLE3 (bf16)"
RUN=G_gptoss_128k_bf16_eagle3_specprobe; env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh golden $HOME/pd_disagg/recipes/gpt-oss_G_128k_bfloat16_eagle3.json $RUN b > $R/runs/$RUN.driver.log 2>&1
if curl -sf -m 5 $U/health >/dev/null; then $PY $S/spec_accept_probe.py --url $U --n 20 --max-new-tokens 256 --out $OUT/mono.jsonl; else log "monolithic server did not come up"; fi
bash $S/pd.sh stop >/dev/null 2>&1; log "phase spec accept compare done"
