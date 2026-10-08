#!/usr/bin/env bash
# Test the F-2 patch (decoder compile policy for Qwen3MoeLLMModel) on host b with a monolithic Qwen3-VL-30B server:
# measure decode speed, check the policy is installed, compare outputs with the eager golden, audit recompiles. Reverts the patch afterwards.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
REPO=$HOME/sglang-torchtpu; PATCH=$HOME/pd_disagg/patches/F-2_qwen3_vl_moe_decoder_compile_policy.patch; B=http://10.128.0.90:30000
log(){ echo "$(date -u +%FT%TZ) $*"; }
log "F2 patch test (rerun with own tier-2 cache name)"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; bash $S/tpu_free.sh || { log "TPU not free, abort"; exit 1; }
cd $REPO && git checkout -- python/sglang/srt/models/qwen3_vl_moe.py && git apply --check $PATCH && git apply $PATCH && log "patch applied: $(git diff --stat | tail -1)" || { log "patch does not apply"; exit 1; }
RUN=G_qwen_128k_fp8_F2patch; OUT=$R/F2_patch; mkdir -p $OUT
rm -rf $HOME/pd_disagg/compile_cache/qwen3-vl_G_128k_fp8_e4m3/golden /dev/shm/torch_tpu_cache/pd_golden_qwen3-vl_G_128k_fp8_e4m3 2>/dev/null; log 'tier-3 and tier-2 caches for this recipe cleared (cold compile)'
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh golden $HOME/pd_disagg/recipes/qwen3-vl_G_128k_fp8_e4m3.json $RUN b > $R/runs/$RUN.driver.log 2>&1
L=$R/runs/$RUN/golden/server.log
if curl -sf -m 5 $B/health >/dev/null; then
  log "server up; policies installed:"; grep -E 'TP0 EP0' $L | grep -iE 'CompilePolicy.*installed|falling back|precompile done|Capture target decode' | head -6 | cut -c1-200
  s=$(grep -m1 -oE '^\[20[0-9-]+ [0-9:]+' $L | tr -d '['); e=$(grep 'ready to roll' $L | grep -oE '^\[20[0-9-]+ [0-9:]+' | tr -d '['); log "cold start with patch: $(( $(date -d "$e" +%s) - $(date -d "$s" +%s) )) s (eager build: 474 s)"
  bash $S/run_golden.sh qwen3-vl $B $OUT 1024,8192,32768,tiny 1 1024
  log "decode speed (TP0 Decode batch lines):"; grep -E 'TP0 EP0.*Decode batch' $L | tail -n 3 | grep -oE 'gen throughput \(token/s\): [0-9.]+' | sed 's/^/   /'
  grep -E 'tokens= 1024' $OUT/run.log | sed 's/^/   /' | cut -c1-120
  log "compiled vs eager golden (c=1), token identity:"
  for Lx in 1024 8192 32768; do $PY $S/golden_diff.py diff --a $R/golden/qwen3-vl_fp8_1m_c1/$Lx.jsonl --b $OUT/$Lx.jsonl --report $OUT/diff_$Lx.json --expect-tokens 1024 2>&1 | grep -E '^isl|ids,' | cut -c1-120; done
  $PY $S/golden_diff.py diff --a $R/golden/qwen3-vl_fp8_1m_c1/tiny.jsonl --b $OUT/tiny.jsonl --report $OUT/diff_tiny.json --expect-tokens 160 2>&1 | grep -E 'ids,' | cut -c1-120
  bash $S/compile_audit.sh $L qwen_F2patch | tee $OUT/compile_audit.txt
  curl -s -m 10 $B/get_server_info > $OUT/server_info.json
else
  log "server did not come up with the patch; last error lines:"; grep -vE 'UserWarning|warnings.warn|Ignore import' $L | grep -E 'Error|error|Traceback|died' | tail -n 6 | cut -c1-220
fi
cp $L $OUT/server.log 2>/dev/null
log "F-4 paired diff (already computed): PD noswamem vs golden noswamem, both c=1:"
for Lx in 8192 32768 131072; do $PY $S/golden_diff.py diff --a $R/golden/gpt-oss_fp8_noswamem_c1/$Lx.jsonl --b $R/A_noswamem/tc1/gpt-oss/$Lx.jsonl --report $R/A_noswamem/golden_diff_noswamem_c1_$Lx.json --expect-tokens 1024 2>&1 | grep -E '^isl|ids,' | cut -c1-120; done
bash $S/pd.sh stop >/dev/null 2>&1
cd $REPO && git checkout -- python/sglang/srt/models/qwen3_vl_moe.py && log "patch reverted; tree clean: $(git status --short | wc -l) changes"
log "phase F2 patch test done"
