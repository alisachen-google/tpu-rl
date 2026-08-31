#!/usr/bin/env bash
set -euo pipefail

PID=""
_sigterm() {
  echo "EXTERNAL-KILL: SIGTERM received at $(date -u +%H:%M:%S); forwarding to workload"
  if [ -n "$PID" ]; then
    kill -SIGTERM "$PID" 2>/dev/null || true
  fi
}
trap _sigterm SIGTERM

MAXTEXT_CONFIG_PATH="${MAXTEXT_CONFIG_PATH:-maxtext/configs/post_train/rl.yml}"

# Export every patch-controlled knob before applying patches. Several patches
# print or capture their value during installation, so ordering is part of the
# reproducibility contract.
export JAX_PLATFORMS="proxy,cpu"
export CONTINUE_DECODE_EOS_CHECK_INTERVAL="${DL_EOS_INT:-0}"
export SYNC_KEEPKV="${DL_KEEPKV:-0}"
export CD_NO_EOS_REDUCE="${DL_NOEOS:-0}"
export JIT_UNROLL="${DL_JITU:-0}"
export SS_PADFIX="${DL_PADFIX:-0}"
export VERIFY_ITEM_TIMEOUT="${DL_VTMO:-5}"
export MAX_SEQ_TOKEN_PER_TPU="${DL_PACK:-0}"
export PHASED_PROFILING_DIR="${ARTIFACT_DIR}/${JOB_IDENTIFIER}/phased_profile"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-gs://ubench-logs/alisachen-tpu7x/compilation_cache/gemma4-e2b-dl}"
export VLLM_ENABLE_V1_MULTIPROCESSING=0

if [ -n "${DL_BUCKET_GAP:-}" ]; then export VLLM_TPU_BUCKET_PADDING_GAP="$DL_BUCKET_GAP"; fi
if [ -n "${DL_DPRR:-}" ]; then export DP_ROUTING=rr; fi
if [ -n "${DL_ENGKW:-}" ]; then export RL_VLLM_ENGINE_KWARGS="$DL_ENGKW"; fi
if [ -n "${DL_SAQ:-}" ]; then export SA_BLOCK_Q="$DL_SAQ"; fi
if [ -n "${DL_SAKV:-}" ]; then export SA_BLOCK_KV="$DL_SAKV"; fi
if [ -n "${DL_RPA:-}" ]; then export RPA_V3_DECODE_BLOCK_SIZES="$DL_RPA"; fi

PATCH_ORDER=(
  patch_vllm_sampler
  patch_cd_proper
  patch_lendist
  patch_rpa_blocks
  patch_cd_masking
  patch_cd_report
  patch_phasetime
  patch_reward_parallel
  patch_sharded_sampling
  patch_profiler_fix
  patch_gc_sync
  patch_packing_knob
  patch_packer_fast
  patch_log_stats
  patch_perf_v2
  patch_dump_completions
  patch_firsttok
  patch_engine_kwargs
  patch_host_extract
  patch_fast_detok
  patch_dp_rr_routing
  patch_sync_keepkv
  patch_cd_noeos
  patch_flash_sa
  patch_jit_unroll
  patch_verify_timeout
)

echo "PATCH-CONTRACT: applying ${#PATCH_ORDER[@]} required patches"
for patch_name in "${PATCH_ORDER[@]}"; do
  patch_file=""
  for candidate in "/workload/patches/$patch_name" "/workload/patches/$patch_name.py"; do
    if [ -f "$candidate" ]; then
      patch_file="$candidate"
      break
    fi
  done
  if [ -z "$patch_file" ]; then
    echo "PATCH-CONTRACT: FAILED missing $patch_name"
    exit 42
  fi

  patch_rc=0
  patch_output=$(python3 "$patch_file" 2>&1) || patch_rc=$?
  printf '%s\n' "$patch_output"
  if [ "$patch_rc" -ne 0 ] || printf '%s\n' "$patch_output" | grep -Eiq \
      'PATCH: FAILED|PATCH: ERROR|^[A-Z0-9-]+: failed|ERROR \(install\)|\(skipped\)'; then
    echo "PATCH-CONTRACT: FAILED $patch_name rc=$patch_rc"
    exit 42
  fi
done
echo "PATCH-CONTRACT: APPLIED ${#PATCH_ORDER[@]}/${#PATCH_ORDER[@]}"

if [ ! -f "$MAXTEXT_CONFIG_PATH" ] && [ -f "src/$MAXTEXT_CONFIG_PATH" ]; then
  echo "MAXTEXT_CONFIG_PATH not found at $MAXTEXT_CONFIG_PATH; using src/$MAXTEXT_CONFIG_PATH"
  MAXTEXT_CONFIG_PATH="src/$MAXTEXT_CONFIG_PATH"
fi
if [ ! -f "$MAXTEXT_CONFIG_PATH" ]; then
  echo "CONFIG-CONTRACT: FAILED MaxText config not found: $MAXTEXT_CONFIG_PATH"
  exit 43
fi

echo "BENCH-CONTRACT: profile=${BENCH_PROFILE:-unknown} tp=${DL_TP:-8} dp=${DL_DP:--1} trainer_fraction=${DL_TFRAC:-0.5} sampler_fraction=${DL_SFRAC:-0.5} steps=${DL_STEPS:-20}"
echo "BENCH-CONTRACT: artifacts=${ARTIFACT_DIR}/${JOB_IDENTIFIER} image is pinned by JobSet"

export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
export PYTHONPATH="/app/src:${PYTHONPATH:-}"
export NUM_PRECOMPILE_WORKERS=1
export NEW_MODEL_DESIGN=1
export JAX_BACKEND_TARGET="${JAX_BACKEND_TARGET:-grpc://127.0.0.1:29100}"

# RLConfig multimodal defaults required by the packing image.
for types_file in /deps/src/maxtext/src/maxtext/configs/types.py /deps/src/maxtext/configs/types.py; do
  if [ -f "$types_file" ] && ! grep -q "Auto-patch RLConfig with Multimodal defaults" "$types_file"; then
    cat <<'MROPE_EOF' >> "$types_file"

# --- Auto-patch RLConfig with Multimodal defaults ---
if 'RLConfig' in globals():
  if not hasattr(RLConfig, 'use_mrope'):
    RLConfig.use_mrope = False
  if not hasattr(RLConfig, 'mrope_section'):
    RLConfig.mrope_section = [24, 20, 20]
  if not hasattr(RLConfig, 'position_id_per_seconds'):
    RLConfig.position_id_per_seconds = 25
MROPE_EOF
    echo "MROPE-SHIM: applied to $types_file"
  fi
done

(
  export ENABLE_PATHWAYS_PERSISTENCE=1
  export ENABLE_PJRT_COMPATIBILITY=true

  python3 -m maxtext.trainers.post_train.rl.train_rl "$MAXTEXT_CONFIG_PATH" \
    model_name=gemma4-e2b \
    data_template_path=maxtext/examples/chat_templates/openmathinstruct2_rl.json \
    chat_template_path=maxtext/examples/chat_templates/gemma-3-27b-chat_template.json \
    tokenizer_path=google/gemma-4-E2B-it \
    dataset_name=nvidia/OpenMathInstruct-2 \
    hf_train_files=hf://datasets/nvidia/OpenMathInstruct-2/data/train_1M-*.parquet \
    train_split=train_1M \
    rl.num_generations=8 \
    rl.grpo_beta=0.0 \
    rl.grpo_epsilon=0.2 \
    rl.epsilon_high=0.28 \
    enable_checkpointing="${DL_CKPT:-True}" \
    checkpoint_storage_use_ocdbt=False \
    checkpoint_storage_use_zarr3=False \
    checkpoint_period=100 \
    async_scheduling="${DL_ASYNC:-False}" \
    chips_per_vm=4 \
    trainer_devices_fraction="${DL_TFRAC:-0.5}" \
    sampler_devices_fraction="${DL_SFRAC:-0.5}" \
    num_batches="${DL_STEPS:-20}" \
    remat_policy="${DL_REMAT:-custom}" \
    attention="${DL_ATTN:-dot_product}" \
    num_test_batches=0 \
    gradient_clipping_threshold=1.0 \
    decode_sampling_temperature=0.8 \
    decode_sampling_top_k=50 \
    decode_sampling_nucleus_p=0.95 \
    max_target_length=8192 \
    max_prefill_predict_length=4096 \
    learning_rate=1e-6 \
    batch_size=256 \
    train_micro_batch_size="${DL_TMBS:-1}" \
    rollout_micro_batch_size=256 \
    rollout_data_parallelism="${DL_DP:--1}" \
    rollout_tensor_parallelism="${DL_TP:-8}" \
    enable_dp_attention=False \
    hbm_utilization_vllm="${DL_HBM:-0.6}" \
    max_num_seqs=256 \
    max_num_batched_tokens=8448 \
    reward_exact_answer=0.0 \
    reward_white_space_format_match=0.0 \
    reward_ratio_guess_to_answer_high=0.0 \
    reward_ratio_guess_to_answer_low=0.0 \
    math_verify_num_procs="${DL_MVPROCS:-32}" \
    scan_layers=False \
    vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
    vllm_additional_config='{"maxtext_config": {"model_name": "gemma4-e2b", "model_call_mode": "inference", "log_config": false, "weight_dtype": "bfloat16", "allow_split_physical_axes": true}, "enable_continue_decode": true, "max_decode_steps": '"${DL_CD:-64}"'}' \
    allow_split_physical_axes=True \
    enable_tunix_perf_metrics=True \
    load_parameters_path=gs://ubench-logs/alisachen-tpu7x/ckpt/gemma4-e2b-raw/0/items \
    skip_jax_distributed_system=True \
    log_period=1 \
    base_output_directory="$ARTIFACT_DIR" \
    run_name="$JOB_IDENTIFIER"
) &
PID=$!

exit_code=0
wait "$PID" || exit_code=$?
echo "MaxText Job Ended at: $(date -u) with exit code $exit_code"

debug_hold_seconds="${BENCH_DEBUG_HOLD_SECONDS:-0}"
if [ "$exit_code" -ne 0 ] && [ "$debug_hold_seconds" -gt 0 ]; then
  echo "DEBUG-HOLD: keeping failed pod alive for ${debug_hold_seconds}s"
  sleep "$debug_hold_seconds"
fi
exit "$exit_code"
