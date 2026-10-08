#!/usr/bin/env bash
# Resume the configuration chain after TPU contention: D, E (gpt-oss, bf16 goldens), then B, C for Qwen3-VL. Ends with the same marker
# the original chain used so the follow-up phases (_phase_hicache_reload.sh) start.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
log(){ echo "$(date -u +%FT%TZ) $*"; }
until bash $S/tpu_free.sh >/dev/null 2>&1; do sleep 120; done; log "TPU free; resuming chain at D"
for cfg in D E; do bash $S/_phase_config.sh gpt-oss $cfg $R/golden/gpt-oss_bf16_c1; done
for cfg in B C; do bash $S/_phase_config.sh qwen3-vl $cfg $R/golden/qwen3-vl_fp8_1m_c1; done
log "phase BCDE done"
