# Full configuration of the cap-8192 parity run `g4lp8k` (2026-10-30)

Extracted from `parity/manifests/g4lp8k-latest-image-cap8192-seed42-parity-250.yaml` (the manifest is the authoritative source).

## Cluster / images

| item | value |
|---|---|
| slice | tpu7x, topology 4x4x4 (64 chips, 8 hosts x 8 chips), Pathways, one JobSet in ns `ubench-regression-tests`, queue `multislice-queue` |
| image | `gcr.io/supercomputer-testing/maxtext-rl-ironwood@sha256:4dc6ca720017bf93b6bae9524519cf5b7a8b16706ca6aab2a355af26d75a6bf3` |
| image | `us-docker.pkg.dev/cloud-tpu-v2-images/pathways/proxy_server:20261028-jax_0.11.0` |
| image | `us-docker.pkg.dev/cloud-tpu-v2-images/pathways/server:20261028-jax_0.11.0` |

## XLA flags (Pathways proxy)

```
--xla_enable_async_all_gather=true --xla_sc_disable_megacore_partitioning=True --xla_tpu_bf16_emission_mode=NATIVE_EMISSION --xla_tpu_dvfs_p_state=6 --xla_tpu_enable_3d_reduce_scatter_decomposer=false --xla_tpu_enable_async_collective_fusion_fuse_all_gather=false --xla_tpu_enable_concurrent_sparse_core_offloading=true --xla_tpu_enable_latency_hiding_layer_scheduler=true --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true --xla_tpu_enable_multi_compute_overlap_in_layer_scheduler=false --xla_tpu_enable_offloading_gather_to_sparsecore=true --xla_tpu_enable_sparse_core_collective_aggregator=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_all_reduce=true --xla_tpu_enable_sparse_core_collective_offload_nd_reduce_scatter=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_num_sparse_cores_for_gather_offloading=1 --xla_tpu_pcie_bandwidth_multiplier=0.03 --xla_tpu_prefer_async_allgather_to_allreduce=true --xla_tpu_scheduler_percent_shared_memory_limit=150 --xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_sparse_core_all_gather_latency_multiplier=1 --xla_tpu_sparse_core_reduce_scatter_latency_multiplier=3 --xla_tpu_use_enhanced_launch_barrier=true --xla_tpu_use_single_sparse_core_for_all_gather_offload=true --xla_tpu_use_tc_device_shape_on_sc=True
```

## `train_rl` arguments (93)

```
model_name='gemma4-e2b'
tokenizer_path='google/gemma-4-E2B-it'
dataset_name='openai/gsm8k'
dataset_processor_path=/workload/patches/gemma_data_parity_native.py
hf_train_files='hf://datasets/openai/gsm8k/main/train-*.parquet'
train_split='train'
rl.num_generations=16
rl.grpo_beta=0.0
rl.grpo_epsilon=0.2
rl.epsilon_high=0.28
checkpoint_storage_use_ocdbt=False
async_scheduling=False
math_verify_num_procs=64
chips_per_vm=8
local_sa_block_q=512
local_sa_block_kv=512
local_sa_block_kv_compute=512
local_sa_block_q_dkv=512
local_sa_block_kv_dkv=512
local_sa_block_kv_dkv_compute=512
local_sa_block_q_dq=512
local_sa_block_kv_dq=512
local_sa_use_fused_bwd_kernel=false
local_sa_q_layout=HEAD_DIM_MINOR
local_sa_k_layout=HEAD_DIM_MINOR
local_sa_v_layout=HEAD_DIM_MINOR
local_sa_fuse_reciprocal=true
eval_local_sa_block_q=512
eval_local_sa_block_kv=512
eval_local_sa_block_kv_compute=512
eval_local_sa_q_layout=HEAD_DIM_MINOR
eval_local_sa_k_layout=HEAD_DIM_MINOR
eval_local_sa_v_layout=HEAD_DIM_MINOR
num_batches=250
free_kv_cache_during_weight_sync=false
vllm_block_size=128
num_test_batches=11
debug=True
profiler=xplane
skip_first_n_steps_for_profiler=100
profiler_steps=1
gradient_clipping_threshold=1.0
decode_sampling_temperature=1.0
decode_sampling_top_k=-1
decode_sampling_nucleus_p=1.0
max_target_length=8704
kv_cache_buffer=8704
max_prefill_predict_length=512
learning_rate=2e-6
eval_dataset_name='openai/gsm8k'
hf_eval_files='hf://datasets/openai/gsm8k/main/test-*.parquet'
eval_split='test'
reward_functions_path=/workload/patches/reward_parity_boxed.py
reward_functions='compute_score'
rl.loss_agg_mode=token-mean
eval_interval=20
eval_sampling_strategy='greedy'
eval_batch_size=128
lr_schedule_type='cosine'
weight_dtype=float32
mu_dtype=float32
num_epoch=5
learning_rate_schedule_steps=250
warmup_steps_fraction=0.04
learning_rate_final_fraction=0.0
adam_eps=1e-8
adam_weight_decay=0.0
batch_size=128
train_micro_batch_size=8
rollout_micro_batch_size=128
data_shuffle_seed=42
init_weights_seed=42
trainer_devices_fraction=1.0
sampler_devices_fraction=1.0
rollout_data_parallelism=128
rollout_tensor_parallelism=1
enable_dp_attention=false
hbm_utilization_vllm=0.3
max_num_seqs=256
max_num_batched_tokens=33280
scan_layers=False
allow_split_physical_axes=True
vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}'
attention='flash'
allow_split_physical_axes=True
enable_tunix_perf_metrics=True
checkpoint_period=20
max_num_checkpoints_to_keep=1000
enable_checkpointing=true
load_parameters_path='gs://ubench-logs/alisachen-tpu7x/ckpt/gemma4-e2b-it-fresh/0/items'
skip_jax_distributed_system=True
vllm_additional_config='{"enable_continue_decode": true, "max_decode_steps": 1024, "maxtext_config": {"model_name": "gemma4-e2b", "model_call_mode": "inference", "log_config": false, "weight_dtype": "bfloat16", "allow_split_physical_axes": true}}'
base_output_directory=${ARTIFACT_DIR}
```

## Environment variables set by the launcher (36)

```
VLLM_BLOCK_SIZE=128
APPROX_TOPK=1
APPROX_TOPK_K=64
VLLM_PREFIX_CACHING=0
DP_ROUTING=rr
DP_SCHED_ROUTING=round_robin
EXTRA_STOP_TOKEN_IDS=[106,50]
FAST_EXTRACT=fast
ASYNC_LOGPROBS=on
MAX_SEQ_TOKEN_PER_TPU=8192
PHASED_PROFILING_DIR=""
PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR=2
CONTINUE_DECODE_EOS_CHECK_INTERVAL="32"
TEXTUAL_STOP=0
VERIFY_ITEM_TIMEOUT=5
REWARD_DUMP_N=512
LENGTH_SHAPE=0
PARITY_SAMPLER_IS=token
PARITY_SAMPLER_IS_THRESHOLD=3.0
ASYNC_LOGPROBS=off
LENGTH_SHAPE_LAMBDA=0.25
LENGTH_SHAPE_TARGET=3500
LENGTH_SHAPE_SCALE=9000
HF_TOKEN=''
HF_HUB_ENABLE_HF_TRANSFER=0
HF_HUB_DISABLE_XET=1
NUM_PRECOMPILE_WORKERS=1
NEW_MODEL_DESIGN=1
TPU_MIN_LOG_LEVEL=0
TF_CPP_MIN_LOG_LEVEL=0
TPU_STDERR_LOG_LEVEL=0
JAX_BACKEND_TARGET=grpc://127.0.0.1:29140
ENABLE_PATHWAYS_PERSISTENCE='1'
JAX_PLATFORMS="${JAX_PLATFORMS:-proxy,cpu}"
ENABLE_PJRT_COMPATIBILITY='true'
ASYNC_REWARD=on
```

## Runtime patches applied at pod start (34, in order)

```
patch_vllm_sampler
patch_cd_proper
patch_lendist
patch_rpa_blocks
patch_cd_masking
patch_cd_report
patch_phasetime
patch_reward_parallel
patch_length_shape
patch_sampler_is
patch_newimg_fix
patch_profiler_fix
patch_gc_sync
patch_packing_knob
patch_packer_fast
patch_block_size
patch_approx_topk
patch_async_detok
patch_async_logprobs
patch_dp_rr_routing
patch_stop_tokens
patch_runner_eos
patch_fast_extract
patch_g4_format_guard
patch_async_reward
patch_jit_unroll
patch_sync_keepkv
patch_host_extract
patch_cd_noglobal
patch_padding_bucket
patch_log_stats
patch_textual_stop
patch_verify_timeout
patch_reward_dump
```

## Files shipped in the patches ConfigMap

```
boxed_math_reward.py
boxed_math_reward_v2.py
gemma4_prs.diff
gemma_data.py
gemma_data_parity.py
gemma_data_parity_native.py
patch_approx_topk.py
patch_async_detok.py
patch_async_logprobs.py
patch_async_reward.py
patch_block_size.py
patch_cd_masking.py
patch_cd_noglobal.py
patch_cd_proper.py
patch_cd_report.py
patch_dp_rr_routing.py
patch_fast_detok.py
patch_fast_extract.py
patch_flash_sa.py
patch_g4_format_guard.py
patch_gc_sync.py
patch_gemma4_prs.py
patch_host_extract.py
patch_jit_unroll.py
patch_lendist.py
patch_length_shape.py
patch_log_stats.py
patch_newimg_fix.py
patch_packer_fast.py
patch_packing_knob.py
patch_padding_bucket.py
patch_perf_v2.py
patch_phasetime.py
patch_profiler_fix.py
patch_reward_dump.py
patch_reward_parallel.py
patch_rpa_blocks.py
patch_runner_eos.py
patch_sampler_is.py
patch_sharded_sampling.py
patch_stop_tokens.py
patch_sync_keepkv.py
patch_textual_stop.py
patch_verify_timeout.py
patch_vllm_sampler.py
reward_parity_boxed.py
```
