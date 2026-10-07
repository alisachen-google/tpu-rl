#!/usr/bin/env bash
# Determinism baseline (rerun goldens at concurrency 1) + GSM8K-100 absolute accuracy on the gpt-oss golden server.
R=$HOME/pd_disagg/results/golden; S=$HOME/pd_disagg/scripts; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
URL=http://10.128.15.217:30000
bash $S/run_golden.sh gpt-oss $URL $R/gpt-oss_fp8_rep2_c1 1024,8192,32768,131072,127999 1 1024
for L in 1024 8192 32768 131072 127999; do echo "=== determinism diff ISL=$L (c2 run vs c1 run)"; $PY $S/golden_diff.py diff --a $R/gpt-oss_fp8/$L.jsonl --b $R/gpt-oss_fp8_rep2_c1/$L.jsonl --report $R/gpt-oss_fp8_rep2_c1/determinism_$L.json --expect-tokens 1024 2>&1 | tail -n 6; done > $R/gpt-oss_fp8_rep2_c1/determinism.txt 2>&1
echo "=== $(date -u +%FT%TZ) GSM8K-100 on golden gpt-oss" > $R/gpt-oss_fp8/gsm8k.log
cd $HOME/sglang-torchtpu && $PY -m sglang.test.run_eval --base-url $URL --model gpt-oss-120b --eval-name gsm8k --num-examples 100 --num-threads 4 --api chat --max-tokens 2048 --temperature 0 >> $R/gpt-oss_fp8/gsm8k.log 2>&1
echo "=== $(date -u +%FT%TZ) done" >> $R/gpt-oss_fp8/gsm8k.log
