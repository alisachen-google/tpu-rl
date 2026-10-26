# gemma4-E2B GSM8K: TPU v7x vs GPU accuracy parity — what the gap was and how it was closed

**Status (2026-10-26):** TPU tracks the frozen GPU reference curve within the ±1 pp standard at
every point measured so far, once both sides are scored with the same scorer. Seed 42 is running to
step 250; seeds 43/44 follow one at a time for the 3-seed gate (step-250 mean v2 accuracy ≥ 91.42,
GPU 92.42).

| step | TPU seed 42 (scorer-v2 acc, 1319 Qs) | GPU 3-seed mean (± SD) | delta |
|---:|---:|---:|---:|
| 0 | **81.50** | 80.31 ± 0.24 | +1.19 |
| 20 | **91.13** | 91.03 ± 0.53 | +0.10 |
| 40 | **90.98** | 91.69 ± 1.07 (seeds 92.27 / 92.34 / 90.45) | −0.71 |

Recipe files: [`parity/`](parity/) (builder, patches, tools, the seed-42 manifest, per-question
eval dumps). Same tree is archived at `gs://ubench-logs/alisachen-tpu7x/parity-recipe/`.

---

## 1. The "gap" was a metric mismatch

The MaxText eval summary line prints two numbers computed by two different scorers:

```
Pre RL Training: corr=1030, total=1319, accuracy=78.09%, ..., mean_reward=0.8224
```

* `accuracy` (= `corr/total`) comes from MaxText's own `score_responses -> check_correctness`
  (its own answer extraction + `math_verify`). It is **not** the GPU metric.
* `mean_reward` is the mean of the configured reward function, which is the frozen GPU scorer
  `boxed_math_reward_v2.compute_score` (exclusive levels 1.0 correct-boxed / 0.1 wrong-boxed / 0.0
  no box). The GPU curve reports v2 **`acc`**.

Because the levels are exclusive, `mean_reward = acc + 0.1 × boxed_but_wrong`, so
`(mean − 0.1)/0.9 ≤ acc_v2 ≤ mean`. Reading the same run in the GPU metric:

| step | MaxText `accuracy` (compared before) | v2 acc bounds from mean_reward | exact v2 acc (new logging) | GPU |
|---:|---:|---|---:|---:|
| 0 | 78.09 | [80.27, 82.24] | 81.50 | 80.31 |
| 20 | 86.96 | [91.13, 92.02] | 91.13 | 91.03 |

The apparent −2.2 / −4.4 pp gaps were MaxText's lenient checker vs the GPU's v2 accuracy.

**Fix:** `patch_sampler_is.py` edit G patches `evaluate_rl.py` to compute scorer-v2 `acc` per
response and log, before every eval summary line:

```
EVAL-BREAKDOWN k=1 total=1319 acc=91.13% boxed_wrong=8.87% nobox=0.00% mean_chars=1335 dump=gs://ubench-logs/<run>/evals/eval_001.jsonl
EVAL-V2ACC total=1319 v2_acc=91.1297% mean_reward=0.9202
```

and writes a per-question JSONL (`i, qhash=sha1(question)[:12], gold, boxed, score, acc, fmt,
n_chars, tail`) so TPU vs GPU can be diffed question by question. `k` is the eval index (step = 20k).

## 2. Real fixes made on the way (all in `parity/`)

| # | What was wrong | Fix | Where |
|---|---|---|---|
| 1 | On-cluster `gemma4-e2b-raw` MaxText ckpt was the **base** model (0–6% GSM8K), `raw2` an over-trained one (94.5%). GPU starts from `google/gemma-4-E2B-it`. | Converted the public `-it` (HF rev 3e22461f) myself → `gs://ubench-logs/alisachen-tpu7x/ckpt/gemma4-e2b-it-fresh/0/items`. Pod-cached HF files hash-match the GPU fixture (`chat_template.jinja`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, `config.json`). | manifest `load_parameters_path` |
| 2 | Token-level truncated importance sampling (GPU: `loss = −A·w·logπθ`, `w = min(exp(clamp(logπθ−logπsampler,−20,20)), 3)`) was inert: the standard tunix GrpoLearner never received `sampler_is` and the rollout never returned logprobs. | tunix **0.1.8** (the pinned image digest `f7c957d0…`, not the `head-612fcd2e` tag = 0.1.7) computes the weights natively; config-only patch adds `sampler_is="token", sampler_is_threshold=3.0` to `GrpoConfig` and `return_logprobs=True` to the rollout. Verified biting in-run: `TIS-ACTIVE weight_mean=1.000 weight_min=0.10–0.40 weight_max≤3.0 frac_clipped≈1e-6` (GPU above-cap fraction 3.7e-6). | `patch_sampler_is.py` A, B, D, E |
| 3 | Eval scored 1280 of 1319 questions (`drop_remainder=True`, 1319 is prime). | `drop_remainder=False` on the eval batch only. | edit C |
| 4 | TIS actor recompute is **unpacked** at `max_target_length=3072`; splash `kv_block=2048` does not divide 3072 → crash after step 0. | `sa_block_kv` default 2048 → 1024 (divides 3072 and packed 8192; tiling only). | `patch_flash_sa.py` |
| 5 | bf16 master weights + bf16 Adam (`rl.yml` default) vs GPU fp32 master + fp32 Adam, bf16 compute. | `weight_dtype=float32 mu_dtype=float32`. Measured effect at step 20: 86.66 → 86.96 (MaxText metric) = noise; kept for recipe fidelity. | `build_g4pt.py` |
| 6 | fp32 `weight_dtype` leaked into the vLLM rollout → `kv_cache.dtype float32 != k/v bfloat16` crash in the RPA kernel. | Rollout `"dtype": "bfloat16"` regardless of trainer master dtype (GPU: bf16 rollout). | edit F |
| 7 | Main thread died at step 40 on GCS `412 PreconditionFailed`: tensorboardX's GCS event-file writer flushes from two threads with a generation precondition. | Tensorboard `log_dir` → local disk (`/tmp/tb_local`); stdout is the parity source of truth. | edit H |
| 8 | Any restart after a mid-run checkpoint died in `maybe_restore`: `Failed to restore from step 20` ← orbax v1 `NoEntryError: incompatible handler` for `model_params`. Resume is broken on this image. | `enable_checkpointing=False` (the inherited command had `enable_checkpointing=true` **after** `checkpoint_period`; pyconfig takes the last occurrence, so flip that line). For an already-running job: delete `<step>/commit_success.txt` (+ `_CHECKPOINT_METADATA`) and orbax reports `Found 0 checkpoint steps` (`tools/guard-ckpt.sh`). | `build_g4pt.py`, `tools/guard-ckpt.sh` |
| 9 | No comparable accuracy metric / no per-question visibility. | `EVAL-V2ACC`, `EVAL-BREAKDOWN`, per-question JSONL (section 1). | edit G |

## 3. Verified identical to the GPU recipe (audited, not assumed)

advantage `(R − mean)/(std_ddof1 + 1e-6)`, group 16, zero-variance groups kept · loss `−A·w·logπθ`,
global token-mean, no PPO clip / KL / entropy · LR 2e-6, warmup 10 steps (`warmup_steps_fraction=0.04`),
cosine → 0 at 250 · AdamW β(0.9, 0.999) eps 1e-8 wd 0 clip 1.0 · batch 128 × 16, T=1, top_k=−1,
top_p=1.0 (explicitly overrides vLLM's generation-config defaults `top_k=64, top_p=0.95`) · system
prompt "You are a helpful assistant. Please reason step by step, and put your final answer within
\boxed{}." · native `<|turn>` template via `apply_chat_template`, exactly one `<bos>` · greedy eval
(`eval_temperature=0.01, top_k=1`) with stop ids `[1, 106, 50]` · scorer-v2 on both sides · GSM8K
test 1319 (`qhash` join key). Also ruled out: truncation at the 2560-token cap (≤ 0.6 % of rollouts).

Step-0 per-question dump (`parity/evals/g4pt1_eval_000.jsonl`): no-box answers (11.1 %) are the
model's own formatting ("**Answer:** 14 lollipops" without `\boxed{}`), not truncation (1 of 147 is
long); RL drives them to 0.0 % by step 20 on both platforms (GPU boxed 87.95 % → 100 %). TPU
completions carry the literal `<turn|>` end marker; the scorer ignores it, strip it before raw diffs.

## 4. How to run the parity check

```bash
cd gemma4/parity
python3 build_g4pt.py 1          # seed 42 -> g4pt1.yaml   (2 -> seed 43, 3 -> seed 44)
CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap
kubectl --context $CTX -n ubench-regression-tests apply --server-side --force-conflicts -f g4pt1.yaml
bash tools/chain-seeds.sh        # runs seeds 43, 44 one at a time after seed 42 completes (ONE 64-chip jobset at a time)
SEEN_MAX=0 bash tools/watch-evals-gcs.sh   # credential-independent: scores each new GCS eval dump locally
```

`build_g4pt.py` derives the manifest from the full-recipe `g4pf.yaml` (not in this tree; the built
`g4pt1-seed42.yaml` is), swaps `patch_sampler_is.py` / `patch_flash_sa.py` into the patches
ConfigMap (the launcher already runs them), sets fp32 master/Adam, disables checkpointing, sets
`maxRestarts: 3`, seeds, and self-verifies (anchors, YAML round-trip, markers).

**Monitoring lines:** `TIS-PARITY PATCH: APPLIED …` · `TIS-ACTIVE weight_min<1` (TIS biting) ·
`EVAL-BREAKDOWN` / `EVAL-V2ACC` (the parity metric) · `Intermediate Eval (step=N)` (MaxText's own,
not comparable).

## 5. Operational pitfalls met during this work

* **Image:** patches must be validated against the pinned digest (tunix 0.1.8), not the tag.
* **Admission:** "pending with free quota" for 30 min is the `check-capacity-prov` node-provisioning
  admission check, not contention; read `.status.admissionChecks`. On 2026-10-26 ~22:14 UTC the
  `ubench-regression-tests` ClusterQueue's `tpu7x-flavor` nominal was reassigned 256 → 0 (every
  tenant pended); read the nominal from the ClusterQueue, never assume 256.
* **One 64-chip jobset at a time** in this namespace; concurrent ones preempt each other.
* **GCS writes** from the dev VM: `gsutil` runs as the compute SA with a read-only scope → 403 and
  `gsutil rm -r` fails silently. Write via the storage client under user ADC or the JSON API with
  `gcloud auth application-default print-access-token`. Reads work, which is what
  `watch-evals-gcs.sh` relies on when the user credential expires (`invalid_rapt`).
* **Automation must treat `kubectl` auth failures as "unknown", never "absent"** — an expired
  credential once made the chain try to launch the next seed (the apply failed on the same error).

## 6. Open items

* Finish seed 42 to step 250; run seeds 43 and 44 (needs tpu7x quota back on the queue).
* GPU per-question validation dumps (runs `gemma4_e2b_gsm8k8k_16n64g_s250_seed{1,2,3}_1024_*`) are
  not in any reachable bucket; staging them under `gs://gke-aishared-gsc-dev/alisachen-gemma4-parity/`
  enables the question-by-question diff against `parity/evals/`.
* Per-step k3_kl / ESS / entropy comparison (GPU: k3_kl 3.4e-4–5e-4, ESS ≥ 0.999, grad-norm median
  0.29) — our tensorboard RL metrics log NaN; stdout `TIS-ACTIVE` covers the IS weights only.
* Exact GPU data order (seed-1 `train_order` parquet is staged; processor `gemma_data_gpu.py` written,
  needs the shuffle disabled in the pod).
