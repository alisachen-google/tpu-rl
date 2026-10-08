#!/usr/bin/env bash
# After the Qwen redo: c=1 Qwen golden for the unaligned 127,999-token prompts (F-9 exact check), then stop.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
log(){ echo "$(date -u +%FT%TZ) $*"; }
until grep -q 'phase A qwen redo done' $R/phase_A_qwen_redo.log 2>/dev/null; do sleep 60; done; log "qwen redo done"
bash $S/pd.sh stop; sleep 5; bash $S/tpu_free.sh || exit 1
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh golden $HOME/pd_disagg/recipes/qwen3-vl_G_128k_fp8_e4m3.json G_qwen_128k_fp8_c1b b > $R/runs/G_qwen_128k_fp8_c1b.driver.log 2>&1
bash $S/run_golden.sh qwen3-vl http://10.128.0.90:30000 $R/golden/qwen3-vl_fp8_1m_c1 127999 1 1024
$HOME/venv/sglang/bin/python $S/golden_diff.py diff --a $R/golden/qwen3-vl_fp8_1m_c1/127999.jsonl --b $R/A/tc1/qwen3-vl/127999.jsonl --report $R/A/tc1/qwen3-vl/golden_diff_c1_127999.json --expect-tokens 1024 2>&1 | tail -n 8
log "GSM8K-100 on the EAGER Qwen golden (reference for the F-2 compiled-vs-eager accuracy check; slow)"
cd $HOME/sglang-torchtpu && $HOME/venv/sglang/bin/python -m sglang.test.run_eval --base-url http://10.128.0.90:30000 --model qwen3-vl-30b --eval-name gsm8k --num-examples 100 --num-threads 8 --api chat --max-tokens 1024 --temperature 0 > $R/golden/qwen3-vl_fp8_1m_c1/gsm8k_eager.log 2>&1; grep -E 'gsm8k_score' $R/golden/qwen3-vl_fp8_1m_c1/gsm8k_eager.log
bash $S/pd.sh stop; log "qwen 127999 golden done"
