# Running Gemma4-E2B GRPO RL on TPU v7x (Ironwood) — Command Reference

Everything needed to launch, monitor, and tear down a gemma4-e2b RL run on
`bodaborg-tpu7x-nap` (project `cloud-tpu-shared-capacity`), as validated during
the 2026-08-17 bring-up. Companion docs: `RUNBOOK.md` (Fix Catalog #1–21),
`GEMMA_BENCH_PLAN.md` (benchmark ladder), `PATHWAYS_BLOCK.md` (jax-version
background).

---

## 0. Prerequisites (already in place)

| Piece | Value |
|---|---|
| Image | `gcr.io/supercomputer-testing/maxtext-rl-ironwood:head-612fcd2e-gemma4` @ `sha256:8d9193…` (= head-612fcd2e + gemma4 vLLM weight mapping, commit `c2c9fbea5`) |
| Checkpoint (Pathways-compatible RAW TensorStore, 541 arrays) | `gs://ubench-logs/alisachen-tpu7x/ckpt/gemma4-e2b-raw/0/items` |
| Chart | `~/tpu_rl/ironwood/recipe-tpu7x/` (values.yaml already points at the image) |
| Launcher | `~/tpu_rl/ironwood/smoke/t6_gemma_launcher.sh` |
| kube context | `gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap` — **always pass `--context` / `--kube-context` explicitly** (a background process on the dev VM flips the default context to the GPU cluster) |

```bash
CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap
cd ~/tpu_rl/ironwood
```

## 1. Pre-flight checks (30 s)

```bash
# Queue still exists? (LocalQueues churn — Fix #17; values.yaml uses `queue: default`)
kubectl --context=$CTX get localqueues

# No leftover pathways pods from a prior release (Fix #12 — port collisions)
kubectl --context=$CTX get pods | grep -E "pathways|worker"
```

## 2. Launch

```bash
RUN=t6g$(date +%H%M)     # any fresh name; never reuse one (Fix #14)
helm --kube-context=$CTX install $RUN recipe-tpu7x/ \
  -f recipe-tpu7x/values.yaml \
  --set-file workload_launcher=smoke/t6_gemma_launcher.sh \
  --set-file workload_config=recipe-tpu7x/config.yml
```

The launcher's train command (the part that matters — full file in
`smoke/t6_gemma_launcher.sh`):

```bash
python3 -m maxtext.trainers.post_train.rl.train_rl "${MAXTEXT_CONFIG_PATH}" \
  model_name='gemma4-e2b' \
  data_template_path=maxtext/examples/chat_templates/openmathinstruct2_rl.json \
  chat_template_path=maxtext/examples/chat_templates/gemma-3-27b-chat_template.json \
  tokenizer_path='google/gemma-4-E2B-it' \
  dataset_name='nvidia/OpenMathInstruct-2' \
  hf_train_files='hf://datasets/nvidia/OpenMathInstruct-2/data/train_1M-*.parquet' \
  train_split='train_1M' \
  rl.num_generations=8 rl.grpo_beta=0.0 rl.epsilon_high=0.28 rl.grpo_epsilon=0.2 \
  enable_checkpointing=True checkpoint_storage_use_ocdbt=False checkpoint_storage_use_zarr3=False \
  async_scheduling=True chips_per_vm=4 \
  num_batches=2 num_test_batches=0 \
  gradient_clipping_threshold=1.0 \
  decode_sampling_temperature=0.8 decode_sampling_top_k=50 decode_sampling_nucleus_p=0.95 \
  max_target_length=8192 max_prefill_predict_length=4096 \
  learning_rate=1e-06 \
  batch_size=8 train_micro_batch_size=1 rollout_micro_batch_size=8 \
  rollout_data_parallelism=-1 rollout_tensor_parallelism=1 enable_dp_attention=False \
  hbm_utilization_vllm=0.6 max_num_seqs=64 max_num_batched_tokens=8448 \
  scan_layers=False \
  vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}' \
  vllm_additional_config='{"maxtext_config": {"model_name": "gemma4-e2b", "model_call_mode": "inference", "log_config": false, "weight_dtype": "bfloat16", "allow_split_physical_axes": true}}' \
  allow_split_physical_axes=True enable_tunix_perf_metrics=True \
  load_parameters_path='gs://ubench-logs/alisachen-tpu7x/ckpt/gemma4-e2b-raw/0/items' \
  skip_jax_distributed_system=True log_period=1 \
  base_output_directory=${ARTIFACT_DIR} run_name=${JOB_IDENTIFIER}
```

### The five gemma4-specific settings (each one is load-bearing)

| Setting | Why (Fix #) |
|---|---|
| `vllm_hf_overrides='{architectures: ["MaxTextForCausalLM"]}'` + `vllm_additional_config.maxtext_config` | gemma4's HF config registers as multimodal → tpu_inference blocks GRPO's prompt logprobs (`tpu_runner.py:1600`). The text-only MaxText-backed sampler avoids it AND enables direct MaxText→MaxText weight sync (no vLLM weight mapping in the path). Mirrors the team's `run_gemma4_e4b_rl.sh` shipped in the image. (#18) |
| **No** `rl.use_agentic_rollout` (non-agentic GRPO) | the agentic env stringifies raw message dicts into the prompt → gemma4-it rambles to the 4096 budget, every rollout dies `MAX_CONTEXT_LIMIT_REACHED`. Upstream bug; non-agentic `process_data` builds prompts correctly. (#19) |
| `data_template_path` + `chat_template_path` (gemma-3-27b) | gemma4-e2b-it needs proper chat turn structure to terminate generations. (#19) |
| `train_micro_batch_size=1` | 262k-vocab logits are NOT batch-sharded on the trainer: tmbs=4 → 130 G HLO temporaries vs ~95 G per Ironwood core-device → OOM at step 1. tmbs=1 matches the team e4b recipe and the GB200 "pd frozen at 2" constraint. (#21) |
| `scan_layers=False` | gemma4 per-layer KV sharing is incompatible with scanned layers (mandatory for this family). |

Do **not** add `hf_name=default` (e4b reference script targets a different
RLConfig schema — Fix #20).

## 3. Monitor

```bash
POD=$(kubectl --context=$CTX get pods --no-headers | awk "/$RUN.*head/{print \$1}")

# live log
kubectl --context=$CTX logs -f $POD -c jax-tpu

# the signals that matter
kubectl --context=$CTX logs $POD -c jax-tpu --tail=-1 | \
  grep -iE "math_verify_pool|train_loss|completed step|pearson|avg_reward|RESOURCE_EXHAUSTED|Traceback|Job Ended"
```

Timeline at 2x2x1 (observed): pods Running ~2 min after admission → ckpt
restore + vLLM engine ~15 min → rewards visible (`math_verify_pool: Processed
N/N items`) ~18 min → `train_step` compile ~20 min → steps complete after.
On failure the launcher prints `DEBUG-HOLD` and sleeps 20 min so logs stay
retrievable — **capture logs before the hold expires**; the cluster reaps
finished jobsets within hours and pod logs are then gone.

Outputs land in `gs://ubench-logs/alisachen-tpu7x/runs/<RUN>/`
(tensorboard, checkpoints at `checkpoint_period` interval).

## 4. Tear down (always drain before the next install)

```bash
helm --kube-context=$CTX uninstall $RUN
until [ -z "$(kubectl --context=$CTX get pods --no-headers | grep $RUN)" ]; do sleep 5; done
```

## 5. Real job run at 4x4x4 (benchmark ladder, GEMMA_BENCH_PLAN.md)

`smoke/t6_gemma_launcher.sh` is the SMOKE (2 steps, batch 8, 2x2x1). The real
run uses **`gemma4_r0_launcher_4x4x4.sh`** (top level, not under smoke/):
identical gemma4-specific settings, benchmark shape `num_batches=30
batch_size=256 rollout_micro_batch_size=256 train_micro_batch_size=1
checkpoint_period=10`, plus the 4x4x4 chart overlay
`recipe-tpu7x/values-4x4x4.yaml` (tpus: 64, instanceType tpu7x:4x4x4,
separate pathways scratch).

```bash
# Requires the admin-provisioned multi-host tpu7x pool (Fix #13 — NAP only creates 2x2x1)
RUN=gemma4-r0-$(date +%m%d-%H%M)
helm --kube-context=$CTX install $RUN recipe-tpu7x/ \
  -f recipe-tpu7x/values.yaml -f recipe-tpu7x/values-4x4x4.yaml \
  --set-file workload_launcher=gemma4_r0_launcher_4x4x4.sh \
  --set-file workload_config=recipe-tpu7x/config.yml
```

P1 OOM scout first: tmbs ∈ {1, 2} ONLY (tmbs=4 is proven-OOM at seq 8192, Fix #21).

## 6. Rebuilding the image (if needed)

```bash
# 2-file overlay on head-612fcd2e (gemma4 weight mapping)
cd /mnt/disks/scratch/g4wm-build && sudo -n docker build -t gcr.io/supercomputer-testing/maxtext-rl-ironwood:head-612fcd2e-gemma4 .
printf '{}' | sudo -n tee /mnt/disks/scratch/docker-root-cfg/config.json >/dev/null
gcloud auth application-default print-access-token | \
  sudo -n docker --config /mnt/disks/scratch/docker-root-cfg login -u oauth2accesstoken --password-stdin gcr.io
sudo -n docker --config /mnt/disks/scratch/docker-root-cfg push gcr.io/supercomputer-testing/maxtext-rl-ironwood:head-612fcd2e-gemma4
```

---

## Debug history (why each knob exists)

| Attempt | Failure | Resolution |
|---|---|---|
| t6g3 | `gemma4-e2b vLLM weight mapping not found` | weight_mapping/gemma4.py + registry (commit `c2c9fbea5`), image `head-612fcd2e-gemma4` |
| t6g3′ | `LocalQueue multislice-queue doesn't exist` | `queue: default` (Fix #17) |
| t6g4 | `prompt_logprobs is not supported for multimodal models` | MaxTextForCausalLM sampler route (Fix #18) |
| t6g5/t6g6 | all rollouts `MAX_CONTEXT_LIMIT_REACHED`, no EOS | non-agentic GRPO + gemma templates (Fix #19); decoded prompt proved agentic path leaks raw dicts |
| t6g7 | `'hf_name' not in …` | drop `hf_name` (Fix #20) |
| t6g9 | trainer OOM: 130.13 G temporaries vs 94.74 G HBM | `train_micro_batch_size=1` (Fix #21) |
| **t6gn8** | **NATIVE ROUTE PASS** (2026-08-18 00:02 UTC): exit 0, 0 unmapped source keys, 2/2 steps, ckpts @ 1+2, EOS-terminating completions — validates weight_mapping/gemma4.py end-to-end (image `head-612fcd2e-gemma4c`: per-layer entries, live-tree destinations, qkv+gate_up fusion via preprocess_src_state, ÷√hidden embedding hook) | native `Gemma4ForCausalLM` sampler |
| t6gn6/t6gn7 | native-route iterations: v1 wildcard keys never matched unscanned state → random sampler weights → gibberish; v2 fixed keys/destinations but missed that ALL layers use fused qkv_proj (65 unmapped) | diagnosis via transfer logs + live-tree analysis |
| **t6g10** | **PASS** (2026-08-17 20:13 UTC): "RL Training Completed Successfully!", exit 0, 0 restarts, 2/2 steps, checkpoints saved at steps 1+2 (`runs/t6g10/checkpoints/actor/`), 39/39 completions math-verified per batch, ~44 min wall incl. compile at 2x2x1 | full config above = the validated gemma4-e2b recipe |
