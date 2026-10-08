#!/usr/bin/env bash
# After the F-11 isolation: accuracy check for the F-2 patch. Warm-start the patched Qwen3-VL server on host b (caches from run 3 are in
# /dev/shm/alisachen_cache), run GSM8K-100, compare with the eager reference, revert the patch, prune the large caches.
S=$HOME/pd_disagg/scripts; R=$HOME/pd_disagg/results; PY=$HOME/venv/sglang/bin/python; export TORCH_DEVICE_BACKEND_AUTOLOAD=0
REPO=$HOME/sglang-torchtpu; PATCH=$HOME/pd_disagg/patches/F-2_qwen3_vl_moe_decoder_compile_policy.patch; B=http://10.128.0.90:30000; OUT=$R/F2_patch
log(){ echo "$(date -u +%FT%TZ) $*"; }
until grep -q 'phase F11 isolation done' $R/phase_F11_isolation.log 2>/dev/null; do sleep 60; done; log "F11 isolation done"
bash $S/pd.sh stop >/dev/null 2>&1; sleep 3; bash $S/tpu_free.sh || { log "TPU not free"; exit 1; }
cd $REPO && git checkout -- python/sglang/srt/models/qwen3_vl_moe.py && git apply $PATCH && log "patch applied"
RUN=G_qwen_128k_fp8_F2patch_acc
env -u TORCH_DEVICE_BACKEND_AUTOLOAD bash $S/pd.sh golden $HOME/pd_disagg/recipes/qwen3-vl_G_128k_fp8_e4m3.json $RUN b > $R/runs/$RUN.driver.log 2>&1
if curl -sf -m 5 $B/health >/dev/null; then
  L=$R/runs/$RUN/golden/server.log; s=$(grep -m1 -oE '^\[20[0-9-]+ [0-9:]+' $L | tr -d '['); e=$(grep 'ready to roll' $L | grep -oE '^\[20[0-9-]+ [0-9:]+' | tr -d '['); log "warm start with patch: $(( $(date -d "$e" +%s) - $(date -d "$s" +%s) )) s"
  log "GSM8K-100 on the COMPILED (patched) Qwen"
  cd $REPO && $PY -m sglang.test.run_eval --base-url $B --model qwen3-vl-30b --eval-name gsm8k --num-examples 100 --num-threads 8 --api chat --max-tokens 1024 --temperature 0 > $OUT/gsm8k_compiled.log 2>&1
  echo "   compiled: $(grep -oE 'gsm8k_score=[0-9.]+' $OUT/gsm8k_compiled.log)   eager: $(grep -oE 'gsm8k_score=[0-9.]+' $R/golden/qwen3-vl_fp8_1m_c1/gsm8k_eager.log 2>/dev/null)"
  bash $S/compile_audit.sh $L qwen_F2patch_acc | head -3
else
  log "patched server did not come up"; grep -hE 'Error|died|Check failed' $R/runs/$RUN/golden/server.log | tail -n 3 | cut -c1-200
fi
bash $S/pd.sh stop >/dev/null 2>&1
cd $REPO && git checkout -- python/sglang/srt/models/qwen3_vl_moe.py && log "patch reverted; tree clean: $(git status --short | wc -l)"
rm -rf /dev/shm/alisachen_cache/tier3/qwen3-vl_G_128k_fp8_e4m3 /dev/shm/alisachen_cache/inductor_dp1_dprank0_tp8; df -h /dev/shm | tail -1
log "phase F2 accuracy done"
