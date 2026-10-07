#!/usr/bin/env bash
R=$HOME/pd_disagg/results
echo "$(date -u +%FT%TZ) launching gpt-oss golden on a"
bash $HOME/pd_disagg/scripts/pd.sh golden $HOME/pd_disagg/recipes/gpt-oss_G_128k_fp8_e4m3.json G_gptoss_128k_fp8 a
bash $HOME/pd_disagg/scripts/run_golden.sh gpt-oss http://10.128.15.217:30000 $R/golden/gpt-oss_fp8 1024,8192,32768,131072,127999 2 1024 &
until curl -sf -m 5 http://10.128.15.217:30000/health >/dev/null; do sleep 20; done
exec env TORCH_DEVICE_BACKEND_AUTOLOAD=0 $HOME/venv/sglang/bin/python $HOME/pd_disagg/scripts/pd_watchdog.py --server a_golden_gptoss=http://10.128.15.217:30000 --interval 10 --hang-secs 300 --out $R/runs/G_gptoss_128k_fp8/watchdog.csv
