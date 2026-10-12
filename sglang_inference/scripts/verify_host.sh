#!/usr/bin/env bash
# Verify a freshly provisioned PD host: env script, venv, checkout, TPU visibility, weights, launcher dry run.
cd "$HOME"
echo "host: $(hostname) user: $(whoami)"
source pd_disagg/scripts/env_singlehost_v7x.sh
echo "checkout: $(git -C sglang-torchtpu rev-parse --short HEAD) $(git -C sglang-torchtpu branch --show-current)"
echo "python: $("$PYTHON" -V)"
"$PYTHON" - <<'PY' 2>&1 | grep -vE '^(I0000|WARNING: All log)'
import torch, sglang, sglang_tpu, torch_tpu
print("sglang from", sglang.__file__)
print("sglang_tpu from", sglang_tpu.__file__)
print("torch", torch.__version__)
import warnings; warnings.simplefilter("ignore")
print("tpu devices:", torch.tpu.device_count())
PY
echo "weights visible: $(ls /mnt/data/tt/hf-cache/hub | grep -c models--)"
R=$(ls pd_disagg/recipes/gpt-oss_G_128k_fp8_e4m3.json 2>/dev/null || ls pd_disagg/recipes/*.json | head -n1)
echo "--- launcher dry run: $R"
cd "$SGLANG_ROOT" && timeout 150 "$PYTHON" -m benchmark.tpu.launch_server --config "@$HOME/$R" --dry-run 2>&1 | grep -vE '^\s*$|^I0000|^WARNING: All log' | tail -n 6 | cut -c1-220
