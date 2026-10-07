#!/usr/bin/env bash
# After the Qwen A phase: (i) 1m-shape goldens on both hosts in parallel (tiny set + 2-chunk set + 256k/512k/1M, c=1),
# (ii) gpt-oss Config A 1m-shape TC-1 (256k/512k/1M/unaligned) at c=1, (iii) F-4 isolation: gpt-oss A with --disable-hybrid-swa-memory.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
LB=http://10.128.15.217:8080; A=http://10.128.15.217:30000; B=http://10.128.0.90:30000
log(){ echo "$(date -u +%FT%TZ) $*"; }
until grep -q 'phase A qwen done' $R/phase_A_qwen.log 2>/dev/null; do sleep 60; done; log "qwen A phase done"
bash $S/pd.sh stop; sleep 5; bash $S/tpu_free.sh || { log "TPU not free, abort"; exit 1; }
# ---- (i) goldens, 1m shape, both hosts in parallel ----
log "goldens 1m shape: gpt-oss on a, qwen on b"
mkdir -p $R/runs/G_gptoss_1m_fp8 $R/runs/G_qwen_1m_fp8
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh golden $HOME/pd_disagg/recipes/gpt-oss_G_1m_fp8_e4m3.json G_gptoss_1m_fp8 a > $R/runs/G_gptoss_1m_fp8.driver.log 2>&1 &
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh golden $HOME/pd_disagg/recipes/qwen3-vl_G_1m_fp8_e4m3.json G_qwen_1m_fp8 b > $R/runs/G_qwen_1m_fp8.driver.log 2>&1 &
wait
log "golden servers up; generating (c=1): tiny, 8192/32768 (2-chunk+), 262144, 524288, 1047552, 255123, 511777, 999999"
( bash $S/run_golden.sh gpt-oss $A $R/golden/gpt-oss_fp8_1m_c1 tiny,32768,262144,255123,524288,511777,1047552,999999 1 1024 ) &
( bash $S/run_golden.sh qwen3-vl $B $R/golden/qwen3-vl_fp8_1m_c1 tiny,1024,8192,32768,131072,262144,255123 1 1024 ) &
wait
log "goldens 1m done"; cp $R/runs/G_gptoss_1m_fp8/golden/server.log $R/golden/gpt-oss_fp8_1m_c1/server.log 2>/dev/null
bash $S/pd.sh stop; sleep 5
# ---- (ii) gpt-oss Config A 1m shape ----
RUN=A_gptoss_1m_run1; log "PD Config A gpt-oss 1m -> $RUN"
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_A_1m.json $RUN 2>&1 | tee $R/runs/$RUN.driver.log
nohup $PY $S/pd_watchdog.py --server prefill=$A --server decode=$B --log decode=$R/runs/$RUN/decode/server.log --interval 10 --hang-secs 600 --out $R/runs/$RUN/watchdog.csv > $R/runs/$RUN/watchdog.out 2>&1 &
bash $S/run_tc1.sh gpt-oss $LB $R/A1m/tc1/gpt-oss 262144,255123,524288,511777,1047552,999999 $R/golden/gpt-oss_fp8_1m_c1 1
for L in 262144 255123 524288 511777 1047552 999999; do python3 $S/needle_check.py --prompts $HOME/pd_disagg/prompts/gpt-oss/$L.jsonl --out $R/A1m/tc1/gpt-oss/$L.jsonl --other $R/golden/gpt-oss_fp8_1m_c1/$L.jsonl --no-fail --report $R/A1m/tc1/gpt-oss/needles_$L.json 2>/dev/null | cut -c1-220; done
log "boundary: ISL = context-1024 accepted, context+1 rejected"
$PY $S/verify_pd_disagg.py --url $LB --router --lengths 1024 --boundary --context-length 1056768 --timeout 3600 2>&1 | tail -n 8 | tee $R/A1m/tc1/gpt-oss/boundary.txt
grep -E 'kv_transfer|transfer' $R/runs/$RUN/decode/server.log | grep -iE 'ms|sec|GB|MB' | tail -n 5 | cut -c1-200
bash $S/compile_audit.sh $R/runs/$RUN/decode/server.log decode_A1m | tee $R/A1m/compile_audit_decode.txt
bash $S/pd.sh stop; sleep 5
# ---- (iii) F-4 isolation: hybrid SWA memory off, 128k shape ----
RUN=A_gptoss_128k_noswamem; log "PD Config A gpt-oss 128k --disable-hybrid-swa-memory -> $RUN"
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh start $HOME/pd_disagg/recipes/gpt-oss_A_128k_noswamem.json $RUN 2>&1 | tee $R/runs/$RUN.driver.log
bash $S/run_tc1.sh gpt-oss $LB $R/A_noswamem/tc1/gpt-oss 8192,32768,131072 $R/golden/gpt-oss_fp8_rep2_c1 1
$PY $S/golden_diff.py generate --prompts $HOME/pd_disagg/prompts/gpt-oss/tiny.jsonl --url $LB --out $R/A_noswamem/tiny.jsonl --max-new-tokens 160 --ignore-eos --return-logprob --concurrency 1 --timeout 900 > $R/A_noswamem/tiny_generate.log 2>&1
$PY $S/golden_diff.py diff --a $R/golden/gpt-oss_fp8_1m_c1/tiny.jsonl --b $R/A_noswamem/tiny.jsonl --report $R/A_noswamem/tiny_diff.json --expect-tokens 160 2>&1 | tail -n 40
$PY $S/golden_diff.py diff --a $R/golden/gpt-oss_fp8_1m_c1/tiny.jsonl --b $R/A/tiny/gpt-oss/tiny.jsonl --report $R/A/tiny/gpt-oss/tiny_diff_vs_golden.json --expect-tokens 160 2>&1 | tail -n 40
log "phase goldens-1m + F4 done"
