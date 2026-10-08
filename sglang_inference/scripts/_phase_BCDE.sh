#!/usr/bin/env bash
# Chain: after the F-11 control pair, run Configs B, C (fp8 goldens), generate bf16 c=1 goldens, run D, E (bf16 goldens) for gpt-oss,
# then B, C for Qwen3-VL. Each config logs to results/phase_BCDE.log and results/<cfg>/<model>/.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
log(){ echo "$(date -u +%FT%TZ) $*"; }
until grep -q 'phase F11 control mr2 done' $R/phase_F11_control_mr2.log 2>/dev/null; do sleep 60; done; log "F-11 control done; starting B/C/D/E chain"
for cfg in B C; do bash $S/_phase_config.sh gpt-oss $cfg $R/golden/gpt-oss_fp8_rep2_c1; done
log "=== bf16 c=1 goldens for D/E (monolithic gpt-oss, bf16 KV)"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3
if bash $S/tpu_free.sh; then
  env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh golden $HOME/pd_disagg/recipes/gpt-oss_G_128k_bfloat16.json G_gptoss_128k_bf16 b > $R/runs/G_gptoss_128k_bf16.driver.log 2>&1
  if curl -sf -m 5 http://10.128.0.90:30000/health >/dev/null; then
    bash $S/run_golden.sh gpt-oss http://10.128.0.90:30000 $R/golden/gpt-oss_bf16_c1 8192,32768,131072 1 1024; tail -n 3 $R/golden/gpt-oss_bf16_c1/run.log | cut -c1-160
  else log "bf16 golden server did not come up"; grep -E 'Error|died|Check failed' $R/runs/G_gptoss_128k_bf16/golden/server.log | tail -n 3 | cut -c1-200; fi
  bash $S/pd.sh stop >/dev/null 2>&1
fi
for cfg in D E; do bash $S/_phase_config.sh gpt-oss $cfg $R/golden/gpt-oss_bf16_c1; done
for cfg in B C; do bash $S/_phase_config.sh qwen3-vl $cfg $R/golden/qwen3-vl_fp8_1m_c1; done
log "phase BCDE done"
