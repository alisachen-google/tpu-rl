#!/usr/bin/env bash
R=$HOME/pd_disagg/results
bash $HOME/pd_disagg/scripts/run_golden.sh qwen3-vl http://10.128.0.90:30000 $R/golden/qwen3-vl_fp8 1024,8192,32768,131072,127999 2 1024 &
until curl -sf -m 5 http://10.128.0.90:30000/health >/dev/null; do sleep 20; done
exec env TORCH_DEVICE_BACKEND_AUTOLOAD=0 $HOME/venv/sglang/bin/python $HOME/pd_disagg/scripts/pd_watchdog.py --server b_golden_qwen=http://10.128.0.90:30000 --log b_golden_qwen=$R/runs/G_qwen_128k_fp8/golden/server.log --interval 10 --hang-secs 300 --out $R/runs/G_qwen_128k_fp8/watchdog.csv
