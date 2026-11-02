# Handbook: gemma4-e2b GRPO on TPU v7x, response cap 8192 (optimized recipe)

How to run the optimized gemma4-e2b reinforcement-learning recipe without access to Google-internal images.
Everything comes from public sources: the MaxText repository (plus one pending fix), public Pathways images,
Hugging Face and the OpenMathInstruct-2 dataset. The recipe itself is the MaxText launch script
`src/maxtext/trainers/post_train/rl/scripts/run_gemma4_e2b_rl.sh` (PR [#6083](https://github.com/AI-Hypercomputer/maxtext/pull/6083)),
and this handbook is the end-to-end procedure around it.

**What you get.** GRPO on OpenMathInstruct-2, 256 prompts × 8 samples per update, prompts up to 4096 tokens and
responses up to 8192 tokens, on one `tpu7x` 4x4x4 slice (64 chips). Validated performance: about 56 s per GRPO update
(median of updates 2–7 with the official `google/gemma-4-E2B-it` conversion), with no runtime patches.

## 0. Before you start

| requirement | notes |
|---|---|
| GKE cluster with a TPU v7x 4x4x4 node pool, Pathways-ready (Kueue + JobSet healthy) | [Create a GKE cluster with Pathways](https://docs.cloud.google.com/ai-hypercomputer/docs/workloads/pathways-on-cloud/create-gke-cluster) |
| Cluster Toolkit CLI `gcluster`, `gcloud`, `kubectl`, Docker (sudoless) | [Running MaxText with Cluster Toolkit](https://github.com/AI-Hypercomputer/maxtext/blob/main/docs/run_maxtext/run_maxtext_via_cluster_toolkit.md) |
| Artifact Registry repository you can push to (Artifact Registry Writer) | the MaxText image you build goes here |
| GCS bucket for checkpoints and outputs | `gs://<your-bucket>/...` |
| Hugging Face account with access to the gated `google/gemma-4-E2B-it` repository | create a token with read access |

```bash
gcloud auth login
gcloud auth application-default login
gcloud auth configure-docker <LOCATION>-docker.pkg.dev
export HF_TOKEN=<your Hugging Face token>
```

## 1. Get MaxText with the KV-sharing fix

gemma4-e2b shares the KV caches of its last 20 layers with earlier layers. Until MaxText PR
[#6081](https://github.com/AI-Hypercomputer/maxtext/pull/6081) is merged, a MaxText image built from `main` produces
gibberish rollouts for this model (GSM8K accuracy 0 %). Build from a checkout that contains the fix:

```bash
git clone https://github.com/AI-Hypercomputer/maxtext.git
cd maxtext
# Option A (after #6081 and #6083 are merged): stay on main.
# Option B (before merge): take the two fixes and the recipe from the fork branches.
git remote add alisachen https://github.com/alisachen-google/maxtext.git
git fetch alisachen
git cherry-pick alisachen/fix/vllm-kv-shared-alias-scatter      # KV-sharing fix (required)
git cherry-pick alisachen/fix/rl-format-regex-backtracking      # reward-regex robustness (recommended)
git cherry-pick alisachen/feat/rl-gemma4-e2b-recipe             # run_gemma4_e2b_rl.sh + tutorial
```

Install the MaxText runner tooling in a virtualenv (this provides `build_maxtext_docker_image`,
`upload_maxtext_docker_image` and `run_tutorial`):

```bash
python3.12 -m venv ~/maxtext-venv && source ~/maxtext-venv/bin/activate
pip install -e '.[runner]'
```

## 2. Build and push the post-training image

```bash
build_maxtext_docker_image WORKFLOW=post-training        # 10–15 min, image name maxtext_base_image
export CLOUD_IMAGE_NAME=<LOCATION>-docker.pkg.dev/<PROJECT_ID>/<REPOSITORY>/maxtext-rl-gemma4
upload_maxtext_docker_image CLOUD_IMAGE_NAME=${CLOUD_IMAGE_NAME?}
export DOCKER_IMAGE=${CLOUD_IMAGE_NAME}:latest
```

The build installs vLLM, tpu-inference and tunix at the commits pinned in
`src/dependencies/extra_deps/post_train_github_deps.txt`. The versions this recipe was validated with are listed in
section 7; newer pins normally work, and the recipe uses only configuration, so nothing in it depends on a patched
dependency.

## 3. Convert the checkpoint

Convert `google/gemma-4-E2B-it` to a MaxText (Orbax) checkpoint. gemma4-e2b must be converted **unscanned**
(`scan_layers=False`, the `gemma4_small` decoder is incompatible with `nn.scan`). The conversion runs on CPU; a VM
with 32 GB of RAM is enough.

```bash
export MODEL_BUCKET=gs://<your-bucket>/ckpt/gemma4-e2b-it
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 -m maxtext.checkpoint_conversion.to_maxtext src/maxtext/configs/base.yml \
    model_name=gemma4-e2b \
    hf_access_token=${HF_TOKEN} \
    base_output_directory=${MODEL_BUCKET} \
    use_multimodal=false \
    scan_layers=False \
    hardware=cpu \
    skip_jax_distributed_system=true \
    checkpoint_storage_use_zarr3=0 \
    checkpoint_storage_use_ocdbt=0 \
    --save_dtype=bfloat16
export MAXTEXT_CKPT_PATH=${MODEL_BUCKET}/0/items
```

`checkpoint_storage_use_zarr3=0` and `checkpoint_storage_use_ocdbt=0` are the Pathways-compatible settings.
Optional sanity check against the Hugging Face model: `tests/end_to_end/tpu/gemma4/e4b/convert_gemma4_base.sh`
shows the `forward_pass_logit_checker` invocation (use `model_name=gemma4-e2b`, `scan_layers=False`).

## 4. Launch

```bash
export PROJECT_ID=<PROJECT_ID>
export CLUSTER_NAME=<CLUSTER_NAME>
export ZONE=<ZONE>                       # e.g. us-central1-b
export BASE_OUTPUT_DIRECTORY=gs://<your-bucket>/rl-runs
export RUN_NAME=gemma4-e2b-rl-$(date +%Y%m%d-%H%M)
export NUM_BATCHES=20                    # 20 updates for a performance check; 500 for a training run
export DOCKER_IMAGE MAXTEXT_CKPT_PATH HF_TOKEN   # from steps 2, 3 and 0

run_tutorial maxtext/trainers/post_train/rl/scripts/run_gemma4_e2b_rl.sh
```

The script submits a Pathways JobSet through `gcluster job submit` with `--compute-type tpu7x-standard-4t`,
`--topology 4x4x4`, pinned Pathways server/proxy images (release `jax_0.11.0`), the XLA flags and the full `train_rl`
command. Override `COMPUTE_TYPE`, `TOPOLOGY`, `PATHWAYS_SERVER_IMAGE` or `PATHWAYS_PROXY_SERVER_IMAGE` through the
environment if your cluster differs.

Monitor:

```bash
gcluster job list
gcluster job logs ${RUN_NAME?} --main-only=false
kubectl get jobset -l gcluster.google.com/workload=${RUN_NAME?}
```

## 5. What a healthy run looks like

- vLLM engine starts within a few minutes (`VLLM_ENGINE_READY_TIMEOUT_S=7200` is only an upper bound).
- Update 1 is slow (compilation): 100–150 s. Updates 2 and later: typically 47–58 s, occasionally 65–78 s when a
  batch contains very long completions.
- Completions are coherent step-by-step math. If every completion repeats one token
  (`trucks trucks trucks …`), the image was built without the KV-sharing fix (section 1).
- TensorBoard metrics under `BASE_OUTPUT_DIRECTORY/<RUN_NAME>/tensorboard`: `rewards/mean`,
  `completions/mean_length` (≈ 450–550 tokens at step 0), `rollout_time`, `step_time_sec`.

Reference (validated 2026-11-02, 8 updates, `google/gemma-4-E2B-it`, p-state 6):

| update | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| seconds | 54.4 | 54.5 | 56.5 | 56.2 | 67.4 | 58.2 |

Median about 56 s per update. Accuracy of the same stack: 3-seed GSM8K 92.39 % on TPU vs 92.42 % with verl on GPU.

## 6. The settings that make it fast (and what not to change)

| setting | value | effect |
|---|---|---|
| `CONTINUE_DECODE_EOS_CHECK_INTERVAL=0` (env) with `enable_continue_decode`, `max_decode_steps=1024` | run each 1024-step decode window on device to full depth | about 30 % faster per update than the default (74.7 s → 53 s in A/B); identical rollouts |
| `DP_SCHED_ROUTING=round_robin` (env) | round-robin request routing over the 128 vLLM data-parallel ranks | avoids probing all ranks per request |
| `max_seq_token_per_tpu=12288` | native trainer packing | required: an unpacked 64-row micro-batch does not shard over 128 devices |
| `max_target_length=12288`, `kv_cache_buffer=12288`, `max_prefill_predict_length=4096` | response cap 8192 | fewer than 0.1 % of completions hit the cap |
| `hbm_utilization_vllm=0.3`, `vllm_block_size=128`, `free_kv_cache_during_weight_sync=False` | KV cache stays allocated across weight syncs | |
| `rollout_data_parallelism=128`, `rollout_tensor_parallelism=1`, `trainer_devices_fraction=sampler_devices_fraction=1.0` | one vLLM rank per device, trainer and sampler share the slice | gemma4-e2b fits on one chip |
| `--xla_tpu_dvfs_p_state=6` | DVFS state used for the reference numbers | |

Things you can change freely: `NUM_BATCHES`, `learning_rate`, `rl.*` GRPO hyper-parameters, `checkpoint_period`,
dataset (`dataset_name`, `hf_train_files`, `data_template_path`). Changing `batch_size`, `rl.num_generations` or the
length settings changes memory use; re-validate `hbm_utilization_vllm` and `max_num_batched_tokens` if you do.

## 7. Versions this recipe was validated with

| component | validated | upstream post-training build (2026-11-02) |
|---|---|---|
| MaxText | `main` of 2026-09-22 + PR #6081 | `main` (needs #6081 until merged) |
| jax / jaxlib / libtpu | 0.11.0 / 0.11.0 / 0.0.44 | per `tpu-post-train-requirements.txt` |
| vLLM | `51da0ca66` (0.28.1rc1) | `d626108b` |
| tpu-inference | `c2289ee4` | `b67ae5f8` |
| tunix | `a3e92fba` (0.1.8) | `385551df` |
| Pathways server / proxy | `jax_0.11.0` release (digests pinned in the script) | same |

The recipe contains no code patches, so later pins are expected to work; if you see a regression, compare against this table.

## 8. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| rollouts are repeated tokens, reward 0 | MaxText without PR #6081 | rebuild the image from a checkout with the fix |
| `Tokenizer ... has no chat_template` or 401 from Hugging Face | gated tokenizer, no token | `export HF_TOKEN=...` before launching (the script passes it as `hf_access_token`) |
| `query.shape[0]=64 ... devices_in_data_fsdp=128` | packing disabled | keep `max_seq_token_per_tpu=12288` |
| updates 40 % slower than the table | continue-decode window exits early | keep `CONTINUE_DECODE_EOS_CHECK_INTERVAL=0` and `enable_continue_decode: true` |
| one update takes many minutes with normal rollouts | degenerate completion in reward regex | include PR #6082 |
| `scan_layers` mismatch at load | checkpoint converted scanned | reconvert with `scan_layers=False` |

## 9. Running without Cluster Toolkit

If you drive Pathways JobSets yourself, the manifest validated for this recipe is
`parity/manifests/g4e2bup-validation-upstream-templates-it-ckpt-8step.yaml` (same `train_rl` arguments and
environment, plus the KV fix applied at container start). Replace the image, GCS paths, namespace and Kueue queue
with yours; keep the `train_rl` arguments and the environment variables as they are.
