# Gemma4-E2B RL Performance Report — TPU v7x (Ironwood) 4x4x4

> **G4R4 MESH CORRECTION (2026-08-31).** g4r4's log (`api_utils` line) shows `data_parallel_size: 16`,
> devices 64-127 (half slice): g4r4 was **tp4×dp16**, NOT tp4×dp32 as recorded below. Same mesh as
> g4r10f. Its good rollout vs g4r10f's collapse = corrupted-era short generations (light KV), not mesh.
> True tp4×dp32 first runs as g4r12r2 (full slice + DL_HBM=0.25 DL_TMBS=4). Also: full-slice serial
> OOMs the train step at default HBM (vLLM's 0.6 reservation leaves 36.7G < 80.3G needed; g4r12r1) --
> half-slice never hits this because trainer devices are disjoint from vLLM's.

Recorded 2026-08-18. All runs: gemma4-e2b (`google/gemma-4-E2B-it` lineage),
GRPO, MaxText+tunix+Pathways, 64-chip / 128-core-device 4x4x4 slice,
bf16 (no quantization), OpenMathInstruct-2. Full head logs:
`gs://alisachen/tpu_rl/ironwood/logs/`. Per-run artifacts (tensorboard,
phasetime JSONs, lendist, checkpoints): `gs://ubench-logs/alisachen-tpu7x/runs/<run>/`.

## Headline — decode-light regime (`g4dl`, GB200-matched invariants)

Full tianyu-rl-14 optimization stack (5 runtime patches + continue-decode
window 64 + sharded logprobs + `async_scheduling=False` + XLA proxy flags
@ scoped_vmem 65536 + compile cache), frozen invariants: 128 prompts × 8
generations = 1,024 completions, β=0, lr 5e-6, temp 0.7, top_k=-1,
format-only reward, caps 512/64, bf16, 20 steps.

| Metric (warm, steps ≥2, n=17) | Value |
|---|---|
| **Global (true-cycle) step time** | **floor 12.4 s · median 14.1 s** · mean 14.2 · max 16.9 |
| Generation phase (engine_generate) | floor 7.0 s · median 8.8 s |
| Rewards (format regex, warm) | 0.73 s |
| Train segment (128×8-microbatch GRPO update + push) | 4.3 s |
| Throughput | **4,635 tok/s** end-to-end · 7,481 tok/s generation-phase |
| Regime proof (LENDIST) | 99.4–100% of completions truncated at cap-64, osl_mean 64.0, isl_mean ~125 (prompt cap never binding) |
| Run health | 20/20 steps, exit 0, zero restarts; step 0 = 1442 s (one-time compiles), step 1 = 93 s (residual warmup), excluded per protocol |

### Vs the reference numbers (same true-cycle definition, same cap-64 batch shape)

| System | Step time | vs g4dl |
|---|---|---|
| **g4dl (this work)** | **14.1 s median** | — |
| GB200 64-GPU (TRL/vLLM colocate, pd=2, hard cap 64) | 122.8 s | **8.7× slower** |
| TPU v7x 64-chip decode-light baseline (94 s floor / 109 s median, ~235 tok/s, rollout ~78% dispatch-idle, pre-CD stack) | 109 s median | 7.7× slower · **~20× lower tok/s** (natural ~11-tok completions) |

Interpretation: the prior TPU baseline's own profiling attributed ~78% of its
rollout to Pathways dispatch idle; the continuous-batching decode window
(fused 64-iteration loop, zero per-token host round-trips) deletes almost all
of it. This is the tianyu qwen3 ladder's thesis transplanted to gemma4 and
amplified by the regime.

Honest caveats: datasets differ across all three systems (ours:
OpenMathInstruct-2; TPU baseline: ads-CTR "letter"; GB200: NuminaMath-CoT);
both references predate the CD/window stack; our sharded-logprobs path changes
the RNG stream by design (distributional, not bitwise, equivalence — LENDIST
band and format-reward sanity were the validity gates, both green).

## Decode-heavy baseline (`g4r0a`, no optimizations — the reference for the stack's value on the real workload)

R0 shape: 30 steps, batch 256×8 = 2,048 completions, 4096+4096 tokens, math
rewards, tmbs=1, CD disabled, no patches/flags. Preempted by Kueue at batch
17 (16 warm cycles captured — sufficient; protocol n≥15).

| Metric (warm, n=15–16) | Value |
|---|---|
| Global cycle | floor 438 s · **median 490 s** (upward drift: completion lengths grow as training progresses; last cycles ~660 s) |
| Rollout phase | floor 284 s · median 357 s (~70% of cycle → rollout-bound) |
| Train+weight-push serial segment | 131 s (extraordinarily stable, ±2 s; contains 256 grad-acc microbatches at tmbs=1) |
| Structure | near-pure serial on-policy: cycle ≈ rollout + train/push (no cross-step pipelining) |

## Perf-PR image A/B (`g4r0p`) — negative result, correctly diagnosed

`-perf` image (upstream PRs #3395/#3401/#1904 merged) with default config:
**no measurable change** (aligned batches 2–3: rollout 312/324 s vs baseline
284/317 s; train gap identical 131 s). Root cause: #3401 requires
`enable_continue_decode` (not set), #3395 requires the RPA env (not set) —
the PRs are activation-gated. This finding shaped the g4dl config, where the
same code, activated, delivered the headline above.

## Decode-heavy stack benchmark (`g4r1c`, 2026-08-21) — the R0 rematch

The full stack (same as g4dl: 5 runtime patches, CD window 64, sharded
sampling, RPA `1,8704,1,2176`, 27 XLA proxy flags, `async_scheduling=False`,
compile cache) on **exactly the g4r0a benchmark shape** (batch 256×8 = 2,048
completions, 4096+4096, math rewards, tmbs=1, 20 steps). Completed 20/20,
zero restarts, records recovered from GCS phasetime/lendist JSONs.

| Metric (median, steps ≥3, n=17) | g4r1c (stack) | g4r0a (baseline) | Gain |
|---|---|---|---|
| **Global step time** | **271.2 s** (floor 253.8, max 441) | 490 s (floor 438) | **1.81×** |
| Rollout (engine_generate) | 145.5 s | 357 s | **2.45×** |
| Trainer segment (global − gen&adv) | 108.1 s | ~131 s | 1.2× |
| Rewards (math_verify, in-step) | 11.9 s (spiky: 4.0–103) | inside rollout gap | — |
| Regime (LENDIST) | 81% median truncation at cap-4096, osl_mean 3412 | same shape | decode-heavy confirmed |

Reading: the stack's value on the *real* workload is a 1.81× cycle speedup,
delivered almost entirely by the rollout rebuild (357→146 s: CD window +
sharded sampling/logprobs + RPA forwarding + XLA flags). The trainer segment
(108 s, tmbs=1 grad-acc ×256) is now the largest single phase → next rungs:
tmbs=2 scout, dp32×tp4. Step 7 (441 s) was a one-off reward-pool fallback
spike (rewards 103 s that step).

### Per-step phase table (tianyu-ladder format; step 1 = cold, median over steps 2–20)

| phase (seconds) | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 | step 11 | step 12 | step 13 | step 14 | step 15 | step 16 | step 17 | step 18 | step 19 | step 20 | median (2–20) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rollout | 606.9 | 132.1 | 218.9 | 135.6 | 205.0 | 140.3 | 167.3 | 229.0 | 141.8 | 142.8 | 146.6 | 145.5 | 144.6 | 147.2 | 143.1 | 148.1 | 142.7 | 145.1 | 149.0 | 150.0 | **145.5** |
| rewards+advantage | 45.6 | 31.0 | 13.5 | 27.7 | 12.6 | 11.2 | 4.3 | 103.6 | 4.8 | 39.7 | 4.5 | 18.6 | 42.7 | 12.2 | 33.2 | 6.0 | 5.3 | 6.9 | 4.6 | 38.6 | **12.6** |
| train+weight sync | 955.7 | 107.6 | 106.5 | 107.9 | 108.2 | 108.1 | 105.9 | 108.4 | 108.3 | 108.9 | 108.1 | 107.3 | 108.4 | 106.9 | 108.1 | 106.7 | 105.8 | 110.9 | 105.2 | 105.3 | **107.9** |
| GLOBAL STEP | 1608.2 | 270.7 | 338.9 | 271.2 | 325.8 | 259.6 | 277.5 | 441.0 | 254.8 | 291.4 | 259.2 | 271.3 | 295.7 | 266.3 | 284.3 | 260.8 | 253.8 | 262.9 | 258.8 | 293.9 | **271.2** |

Phase mapping vs the qwen3 ladder tables: our `rollout` = engine_generate
(vLLM call incl. detokenize); `rewards+advantage` = generate_and_advantage −
engine_generate (math_verify + advantage + marshal — the ladder folds this
into its "idle" row); `train+weight sync` = global − generate_and_advantage.
Step 1 is the cold step (compiles), excluded from the median.

| distribution (steps 2–20 pooled, n=38,912) | mean | p50 | p90 | p99 | max |
|---|---|---|---|---|---|
| ISL (prompt tokens) | 132.8 | 118 | 170 | 389 | 2691 |
| OSL (generated tokens) | 3332.1 | 4096 | 4096 | 4096 | 4096 |

OSL p50 already at the 4096 cap (~81% truncation) — deep decode-heavy regime,
vs the qwen3 ladder's 8192-cap runs (OSL p50 2576, ~23–26% truncation): per
generated token our regime is denser, per step qwen generates ~11% more tokens.

### Getting it to run at all: Fixes #23–#24 (three attempts died at the same spot)

- **Fix #23 (eviction guard)**: ClusterAutoscaler deleted the NAP 4x4x4 pool
  under g4r1 attempt 1 (`ClusterAutoscalerCleanup node/... removed` events).
  `cluster-autoscaler.kubernetes.io/safe-to-evict: "false"` on head+worker
  pod templates.
- **Fix #24 (RM OOMKill)**: g4r1/g4r1b attempts died ~25 min into the
  train-step compile with clean SIGTERMs. Cloud Logging node stream showed
  `Memory cgroup out of memory: Killed process (cloud_pathways_)
  anon-rss:15537816kB` — the Pathways **resource manager** hosts the
  server-side compile, and gemma4's unscanned 262k-vocab train HLO peaks
  >15.5 G against the 16 G container limit. Deterministic at end-of-compile.
  Fix: RM memory limit 16G→48G (+ cpu 4→8, which also cut the compile from
  ~25 to ~13 min). Sampled RM peak on the passing run: 13.7 G (2-min sampling).
  Forensics recipe recorded: container stdout (what the app saw) + k8s_node
  logs (what the kernel did) + pod events/jobset controller (who ordered the
  teardown) — all in Cloud Logging, surviving pod deletion.

### Gemma4 vs Qwen3 on the same 4x4x4 — comparison and analysis

Both runs: 64-chip tpu7x 4x4x4, Pathways, GRPO, 2,048 completions/step
(256×8), full tianyu optimization stack, medians over warm steps.

| | qwen3-0.6b (`tianyu-rl-17a`) | gemma4-e2b (`g4r1c`) |
|---|---|---|
| Global step (median) | **133.6 s** | **271.2 s** |
| Rollout | 105.4 s | 145.5 s |
| Trainer segment | ~25.1 s | 108.1 s |
| Rewards (in-step) | 2.8 s | 12.6 s (spiky 4–103) |
| Completion cap / truncation | 8192 / ~23% | 4096 / ~81% |
| Tokens generated per step | ~7.6 M (osl_mean 3708) | ~6.8 M (osl_mean 3332) |
| **Rollout throughput** | **~72k tok/s** | **~48k tok/s** |
| tmbs (grad-acc microbatch) | 32 (8 microbatches) | 1 (256 microbatches) |
| CD window / EOS interval | 128 / 8 | 64 / 1 |
| Model route | native `Qwen3ForCausalLM` | MaxText-backed (`MaxTextForCausalLM`) |
| Rewards recipe | math (check_numbers ~0.39) | format-only weights (math_verify still runs) |

Where the 2× gap comes from:

1. **Mostly legitimate model cost (rollout).** Per generated token gemma is
   only ~1.5× slower (48k vs 72k tok/s) for a model with ~3–4× the compute
   per token (35 unscanned layers vs 28 scanned, head_dim 256/512 vs 128,
   262k vs 152k vocab, per-layer embedder + KV-share lookups). Rollout-side
   the gemma port is running efficiently.
2. **Trainer microbatching is the biggest avoidable gap (108 s vs 25 s).**
   Gemma is pinned at `train_micro_batch_size=1` (Fix #21: the RL trainer
   materializes full 262k-vocab per-token logprobs; tmbs=4 OOM'd at 130 G).
   Qwen runs tmbs=32. The upstream pretrain recipe (tpu-recipes gemma4-2b
   8k-bf16-tpu7x-4x4x4) fits `per_device_batch_size=2` at seq 8192 on the
   same hardware — pretraining's fused cross-entropy never materializes the
   logits. Structural headroom: adopt chunked/fused logprob computation in
   the RL trainer; tactical: the tmbs=2 scout (g4s2).
3. **Decode dispatch settings differed** — qwen: CD window 128 + EOS-check
   interval 8; gemma: 64/1. Tested in scout g4s1: deepening gemma to 128/8
   did NOT improve rollout (~150 s vs 145.5) and inflated reward-verify
   cost — gemma's rollout at this shape is not dispatch-bound; window 64
   already amortizes the host round-trips.
4. **Regime density**: 81% of gemma completions hit the 4096 cap (OSL p50 =
   cap) vs 23% for qwen at 8192 — gemma's decode batch stays maximally
   full, so its per-step token volume is only ~11% lower despite the halved
   cap.
5. **Reward compute**: qwen's stays at ~2.8 s (parallel pool); gemma's
   math_verify on longer truncated math text is spiky (4→103 s) and is the
   main step-to-step variance source — smoothing it is worth ~15 s off the
   median.

Validity of the comparison: both runs 20/20 steps, exit 0, zero restarts;
qwen rewards flat in the pre-registered 0.13–0.16 band; gemma rewards rising
0.017→0.027 with textbook grad-norm decay (337→6) and kl=0 at β=0. Reward
*recipes* differ (math-weighted vs format-only) — a second-order effect on
step time since math_verify executes in both; noted as a caveat, closable by
rerunning g4r1c with g4r0a's exact reward weights.

### Side result — qwen3 ladder point run for tianyu (run-17 → `tianyu-rl-17a`)

Teammate's RPA decode-tile sweep point 2 (`1,1024,1,1024`), their manifest
verbatim (renamed 17a; eviction guard added): 20/20 steps, exit 0, all bands
green. **Official read: 133.6 s median global (steps ≥3) vs parent 15b's
126.1 s — the 1024 fetch/compute tile is a ~6% regression, entirely
rollout-side (92.1→105.4 s), train unchanged (24.8→25.1 s).** Matches the
pre-registered DMA-bandwidth risk; ladder keeps `1,16384,1,4096`.
Artifacts: `gs://ubench-logs/tianyu-rl-17a/`.

## NEW CHAMPION — `g4r2` (2026-08-22): full stack + tmbs=8 on the R0 shape

20/20 steps, exit 0, zero restarts. Same config as g4r1c except
`train_micro_batch_size=8` (the sweep's winner).

| Metric (median, steps ≥3, n=17) | g4r2 | g4r1c | g4r0a baseline |
|---|---|---|---|
| **Global step time** | **169.2 s** (floor 163.1) | 271.2 s | 490 s |
| Rollout | 146.8 s | 145.5 s | 357 s |
| Rewards+advantage | 6.0 s | 12.6 s | — |
| Train+weight sync | **16.4 s** | 107.9 s | ~131 s |
| **vs baseline** | **2.90×** | 1.81× | — |

Per-step globals (step 1 cold 1512 s excluded): 180 248 169 230 163 188 258
169 165 168 181 193 185 173 168 168 169 168 169 — steady-state band
163–193 s with two long-rollout excursions (230/258 s).

Cross-model context: qwen3-0.6b on the same stack and chips runs 133.6 s;
gemma4-e2b — a ~3–4× costlier model per token — is now within 27% of it.
The cycle is rollout-bound (87% of the median step); remaining rungs are
rollout-side (RPA tile sweep, dp32×tp4) and reward-spike smoothing.

### 1. Exact change list — g4r2 vs g4r1c (the previous champion)

**One workload change and nothing else**: `train_micro_batch_size` 1 → **8**
(env `DL_TMBS=8`; 256-sequence batch now runs as 32 grad-acc microbatches
instead of 256). Byte-identical between the two runs:

- Image: `gcr.io/supercomputer-testing/maxtext-rl-ironwood:head-612fcd2e-gemma4c-perf`
- Launcher: `gemma4_e2b_r1_launcher.sh` (same file; tmbs is env-parametrized)
- Runtime patches: patch_vllm_sampler, patch_cd_proper, patch_lendist,
  patch_phasetime, patch_sharded_sampling (+ patch_dump_completions mounted)
- Rollout config: DL_TP=8 (dp16×tp8), `RPA_V3_DECODE_BLOCK_SIZES=1,8704,1,2176`,
  CD window 64 (EOS-check interval default 1), `async_scheduling=False`,
  hbm 0.6, max_num_batched_tokens 8448, max_num_seqs 256
- Workload: batch 256×8 = 2,048 completions, 4096+4096, format-weight rewards
  (math_verify executes), lr 1e-6, 20 steps, `checkpoint_period=100`,
  same JAX compile cache dir
- Pathways: 27 proxy XLA flags @ scoped_vmem 65536, RM 48G/8cpu (Fix #24),
  eviction guard (Fix #23)

Non-measurement infra deltas: ran on chart-c (ports 29200s) instead of
chart-b — both charts carry Fixes #23/#24 — under a fresh release name
(avoids the tensorboardX GCS-412 collision between jobset attempts).

Evidence licensing tmbs=8 (scouts, 8 steps each, single-variable):
g4s2b tmbs=2 → trainer 55.5 s; g4s4b tmbs=4 → 29.7 s; g4s8 tmbs=8 → 15.8 s;
zero trainer OOMs — perfect halving per doubling; prior: the tpu-recipes
gemma4-2b pretrain recipe fits per_device_batch_size=2 @ seq 8192 on the
same hardware.

Launch command:

```bash
helm install g4r2 recipe-tpu7x-c/ \
  -f recipe-tpu7x-c/values.yaml -f recipe-tpu7x/values-4x4x4.yaml -f recipe-tpu7x-c/values-dl.yaml \
  --set workload.image=gcr.io/supercomputer-testing/maxtext-rl-ironwood:head-612fcd2e-gemma4c-perf \
  --set workload.envs[4].name=DL_STEPS --set workload.envs[4].value=20 \
  --set workload.envs[5].name=DL_TP    --set workload.envs[5].value=8 \
  --set workload.envs[6].name=DL_RPA   --set workload.envs[6].value=1\,8704\,1\,2176 \
  --set workload.envs[7].name=DL_TMBS  --set workload.envs[7].value=8 \
  <5 runtime patches via --set-file workload_patches.*> \
  --set-file workload_launcher=gemma4_e2b_r1_launcher.sh \
  --set-file workload_config=recipe-tpu7x-c/config.yml
```

### 2. g4r2 per-step phase table (step 1 = cold, excluded from medians)

| phase (seconds) | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 | step 11 | step 12 | step 13 | step 14 | step 15 | step 16 | step 17 | step 18 | step 19 | step 20 | median (2–20) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rollout | 626.7 | 132.0 | 222.1 | 137.4 | 207.5 | 142.4 | 167.1 | 234.3 | 146.8 | 144.1 | 146.1 | 144.1 | 148.1 | 148.2 | 150.5 | 148.8 | 146.1 | 145.8 | 145.6 | 147.3 | **146.8** |
| rewards+advantage | 47.1 | 32.7 | 8.7 | 15.3 | 5.6 | 4.4 | 4.7 | 7.8 | 6.1 | 4.0 | 4.7 | 18.7 | 28.5 | 20.4 | 6.0 | 3.8 | 5.4 | 7.8 | 5.8 | 5.8 | **6.0** |
| train+weight sync | 838.6 | 15.8 | 17.4 | 16.4 | 16.8 | 16.2 | 16.1 | 16.2 | 16.2 | 17.2 | 17.4 | 18.2 | 16.6 | 16.6 | 16.2 | 15.8 | 16.5 | 15.9 | 16.4 | 16.1 | **16.4** |
| GLOBAL STEP | 1512.5 | 180.5 | 248.3 | 169.1 | 229.9 | 163.1 | 187.9 | 258.2 | 169.2 | 165.3 | 168.2 | 181.0 | 193.2 | 185.2 | 172.7 | 168.5 | 167.9 | 169.4 | 167.7 | 169.1 | **169.4** |

Official read (protocol: steps ≥3, n=17): **global median 169.2 s · floor
163.1 · mean 181.5**. Trainer segment is metronomic (15.8–18.2 s across all
warm steps). Rollout alternates the usual two modes (~146 s vs 208–234 s on
4 of 19 warm steps). Rewards stayed tame this run (median 6.0 s, worst 33 s
— no 100 s+ fallback spike). LENDIST: truncation 63.7→84.1% (cap 4096),
osl_mean 2730→3523 — the standard within-run completion-length drift as
training progresses, matching g4r1c's regime.

### 3. Run logs and artifacts — links

| Run | Head log | Per-step records (phasetime / lendist / tensorboard / ckpts) |
|---|---|---|
| **g4r2 (champion)** | `gs://alisachen/tpu_rl/ironwood/logs/g4r2-head.log` | `gs://ubench-logs/alisachen-tpu7x/runs/g4r2/` |
| g4r1c (prev champion) | `gs://alisachen/tpu_rl/ironwood/logs/g4r1c-head.log` (recovered from Cloud Logging) | `gs://ubench-logs/alisachen-tpu7x/runs/g4r1c/` |
| g4s2b (tmbs=2 scout) | `gs://alisachen/tpu_rl/ironwood/logs/g4s2b-head.log` | `gs://ubench-logs/alisachen-tpu7x/runs/g4s2b/` |
| g4s4b (tmbs=4 scout) | `gs://alisachen/tpu_rl/ironwood/logs/g4s4b-head.log` | `gs://ubench-logs/alisachen-tpu7x/runs/g4s4b/` |
| g4s8 (tmbs=8 scout) | `gs://alisachen/tpu_rl/ironwood/logs/g4s8-head.log` | `gs://ubench-logs/alisachen-tpu7x/runs/g4s8/` |
| g4s1 (CD-128 regression) | `gs://alisachen/tpu_rl/ironwood/logs/g4s1-head.log` | `gs://ubench-logs/alisachen-tpu7x/runs/g4s1/` |
| g4r0a (baseline) | `gs://alisachen/tpu_rl/ironwood/logs/g4r0a-head.log` | `gs://ubench-logs/alisachen-tpu7x/runs/g4r0a/` |

All links live (scout logs uploaded 2026-08-24). Any reaped pod's full log
is also recoverable from Cloud Logging: `resource.type="k8s_container" AND resource.labels.pod_name="<pod>"
AND resource.labels.container_name="jax-tpu"` in project
cloud-tpu-shared-capacity. Launch bundle + launchers + charts:
github.com/alisachen-google/tpu-rl (`gemma4/`, `chart/`).

## Device vs Host vs Collective time — per decode window, by mesh (measured)

Per-window profile breakdown (host trace start-to-start + device xplane), showing
rollout is host-bound at every mesh; the mesh win is host-side (fewer engines).

| | tp=1 (g4r7b, dp64) | tp=2 (g4r10d, dp32) | tp=4 (g4r4, dp32) |
|---|---|---|---|
| Host window (wall) | 780 ms | 993 ms | 1014 ms |
| Device (jit__decode_core_impl) | 14.9 ms | 31.6 ms | 34.8 ms |
| -- collectives | 7.8 ms (52% of device) | 13.5 ms (43%) | 14.6 ms (42%) |
| -- composition | permute 7.35 | all-reduce 7.8 + all-gather 4.6 | all-reduce 10.8 + all-gather 3.7 |
| Device / host util | 1.9% | 3.2% | 3.4% |
| Collectives / host wall | 1.0% | 1.4% | 1.4% |

Reading: host dominates (97-98% of wall) everywhere. tp=1 pays `collective-permute`
(sharded-path index shuffles); tp>=2 pays real TP `all-reduce` + `all-gather` (the
latter is the full-vocab FALLBACK sampler -> padfix target, ~4.6ms at tp=2).
Collectives are ~1.4% of wall, so padfix trims device not wall. NOTE g4r4 is the
old perf-image/corrupted-gen tp=4 (device/collective SHAPE reference only; its 128.7s
end-to-end is a different regime from the recent g4r10f tp=4 ~280s/step).

Profile files (xprof/perfetto double-check), under
gs://ubench-logs/alisachen-tpu7x/runs/<run>/phased_profile/decode_only/dp_rank_0/plugins/profile/<ts>/:
- tp=1 g4r7b 2026_08_27_12_05_59: host gke-...mg47.trace.json.gz ; device server-...rppc.0.0.xplane.pb
- tp=2 g4r10d 2026_08_31_01_20_16: host gke-...ounk.trace.json.gz ; device server-...1sw8.0.0.xplane.pb
- tp=4 g4r4 2026_08_25_08_32_00: host gke-...mg47.trace.json.gz ; device server-...rjsg.0.0.xplane.pb

## BREAKTHROUGH: tp sweep on corrected generation (2026-08-31)

The bisect's env knobs only trimmed edges; the MESH is the real lever. On the
owned-quota corrected baseline (g4r9x stack, 6-step):

| Mesh | rollout med | global med | vs champion 125.8 |
|---|---|---|---|
| tp=1 x dp=64 (our "champion") | 98.6 | 127 | baseline |
| tp=2 x dp=32 (g4r10d) | 93.6 | **115.1** | **-11 s, NEW BEST** |
| tp=4 x dp=16 (g4r10f) | RUNNING | -- | -- |

WHY tp=2 wins (contradicts the naive roofline): tp=2 halves the KERNEL (AI 15->7.5
+ adds collectives -> worse compute) BUT halves DP (64->32), so the single-process
head drives 32 engines instead of 64 -> HALF the head-loop fan-out. Rollout is
host-bound, so fewer engines beats better kernel. Gap-A win via fewer DP ranks.
=> tp=1 was NOT optimal; higher TP (fewer engines) helps because the head is the
bottleneck. Testing tp=4 (dp=16) = tianyu's mesh.

CORRECTION to the divergence analysis: verified tianyu-rl-143 launcher configmap
(ground truth). tianyu runs qwen3-0.6b on nvidia/OpenMathInstruct-2 (SAME dataset
as us), SINGLE-TURN (no use_agentic_rollout -- earlier "agentic multi-turn" claim
was WRONG, sourced from the vaneja recipe misattribution), TP4xDP32,
trainer_devices_fraction=1.0, CD window 128, hbm 0.3, verify procs 64, attention
flash. So the real qwen3-vs-us difference is MESH (TP4 vs TP1) + model (qwen3 vs
gemma4) -- NOT workload style. This is why moving toward their higher-TP mesh helps
us. At tp>1 the vocab-sharded sampler activates -> padfix (their #3, -9.5s) becomes
relevant; g4r10d/f run without it, so tp>1 may improve further with padfix.

## Bisect campaign on corrected generation (2026-08-30/31) — owned-quota lane

ACCESS: unblocked the preemption problem by joining `cmcs-ai-ninja-team@twosync.google.com`
(grants power-users RBAC in ns `ubench-regression-tests`, ClusterQueue owns 256 chips).
All runs below submit there via `--namespace ubench-regression-tests --set namespace=...`
at priority 500 (medium) -- unpreemptable, vs the ~11 preempted borrow attempts in `default`.
CHECKPOINT BUG found + worked around: a stale checkpoint from a preempted same-run_name
attempt makes orbax restore crash on typed PRNG keys (`ValueError: Unsupported dtype:
key<fry>`); probes now run `DL_CKPT=False` (fresh run_name also avoids it). Champion loads
raw params so is unaffected.

METHOD: one variable per run vs a fixed baseline, judge on median AND mean of steps>=3
(mean matters -- tails dominate real training throughput), fast-fail on a clear regression.

### The reference baseline -- g4r9r (champion + bucket fix only)
rollout 99.1/98.9/98.6/98.6 (flat, median 98.6), global ~127-132. == champion g4r7b
(96.7/125.8) with CORRECTED generation. Confirms two things: (1) the prefill-bucket
corruption fix (VLLM_TPU_BUCKET_PADDING_GAP=32768) is perf-NEUTRAL as well as correct
(echo 72%->39%) -- ADOPT UNCONDITIONALLY; (2) the always-on patches host_extract
(vectorized EOS trim) + fast_detok (batched detokenize) are perf-neutral (folded in free).
Mesh confirmed tp=1 x dp=64 (NOT dp=128 -- prior mislabel; data_parallel_size=64).

### Results

| Run | Change vs baseline | Rollout med / mean | Verdict |
|---|---|---|---|
| g4r9p1 | full host bundle (prefix-off + rr + keepkv + trim + detok + 64proc) | ~209 steady | REJECT (net +110s) |
| g4r9s | + prefix_caching=false ONLY | 77.6 / 120.7 | REJECT: median win, MEAN regression |
| g4r9t | + rr-routing ONLY | ~214-219 | REJECT: ~2x rollout (the bundle's main regressor) |
| g4r9w | + keepkv+64proc+noeos+FLASH | crash | flash_sa shim missing local_use_splash_scheduler; flash DEFERRED |
| g4r9x | + keepkv+64proc+cd_noeos | RUNNING | -- |

WHY prefix-off rejected: removes the DP router's per-request all-rank cache probe (helps
the single-core head on light batches -> ~76s) BUT the 8 identical generations/group no
longer share prefix KV, so on some batches the KV pool saturates -> capacity-limited
wave scheduling -> rollout spikes to ~220s. Median 77.6 (win) but mean 120.7 (loss vs
99). Not length-correlated (isl/osl flat across spike vs clean steps). Fat tail = net loss
for real (summed) training time.

WHY rr-routing rejected: round-robin ignores load, packs ranks unevenly; the slowest rank
straggles and the lockstep-SPMD rollout waits on it. Stock probing router earns its cost by
keeping ranks balanced. Confirmed on same-seed batches: baseline 99s -> rr 214/219s.

### The reframe: the enemy is TAILS, not steady state
Clean steps already hit global ~97-102s (BELOW champion 125.8). What inflates the median
(g4r9s global median 171) is spikes in TWO phases on DIFFERENT steps:
- rollout spikes (~220s) on heavy/KV-pressured batches (prefix-off makes these worse)
- REWARD spikes: rewards median 5.6s but steps hit 80s and 211s (math_verify choking on
  certain completions) -- a NEW, separate, large lever. g4r9x tests 64 verifier procs on it.
Trainer is a non-issue: 16.8s, zero step-to-step variance.

### Roofline verdict on remaining headroom
Decode kernel is ON the memory roof (109 TFLOP/s ~= 100% of AI=15 ceiling; ~4.7% MFU is
all AI permits). Rollout runs at ~1% of that kernel because the head drives 64 engines on
1-2 of 44 cores (VLLM_ENABLE_V1_MULTIPROCESSING=0). Env-knob bisect is nearly exhausted and
only trims edges; the real ~2-3x is Gap A -- the single-process head. Next lever (code, not
env): process-pool _process_continue_decode_outputs and/or revisit multiproc engines under
Pathways. Kernel/tile and trainer are closed (roofline-proven).

## Roadmap after the qwen3-transfer ladder (agreed 2026-08-30)

Queue: g4r9p1 (host bundle + bucket fix, 10-step probe; preempt-cycling under
the ubench-regression cohort, maxRestarts=50) -> g4r9p2 (p1 + cd_noeos +
flash trainer attention q1024/kv2048) -> 20-step confirm of the winning
stack. Then the three Meta-recipe adoptions, in order:

1. Threefry replay-exact graft (from tpu-inference PR #3100, unmerged):
   regenerate Gumbel noise at candidate GLOBAL vocab indices inside
   patch_sharded_sampling -> bit-exact sampling vs jax.random.categorical.
   Payoff is methodological: byte-identical A/B gates for all future
   sampler/mesh/async experiments (g4r8's verdict would have been instant).
2. Stop-token hygiene (zero-risk launcher default): rollout stop tokens
   eos=[1,106,50] + skip_special_tokens=True for gemma free-text rewards.
3. OPD / split-pool overlap probe (tunix AgenticGRPOLearner): overlap
   rollout with train at one-step off-policyness. Our 0.5/0.5 mesh already
   matches the OPD topology. NOTE: changes the algorithm contract
   (one-step off-policy) -- run as an explicitly-labeled variant, not a
   drop-in champion candidate, and gate on reward-curve equivalence.

Warnings owed to the Meta/gemma4 recipe authors: async_scheduling=true is
in their launch command (our g4r8: VOID, sampling-stream change under
continue-decode); the prefill-bucket corruption (they are safe only because
8192x8 lands on a power of two); "262k vocab forces TP=8" is
shape-dependent, not fundamental (our TP=1 champion).

## Gap-B experiments (2026-08-28): async_scheduling VOID; window-128 running

Decode-phase trace analysis (g4r7b phased_profile) found rollout's per-window
anatomy: 780 ms host wall containing a 546 ms BLOCKING jax.device_get
completion-wait around 15 ms of TPU compute (~2% device utilization during
decode; full derivation + span stacks in the g4r8 explainer artifact).
Two levers follow ("Gap B" = per-window host tax; "Gap A" = cross-engine
packing on the shared head, unproven pending a head-wide capture).

### g4r8 — async_scheduling=True: VOID (contract violation) + no win

The image's tpu_runner has a complete async continue_decode path
(copy_to_host_async + deferred collection in _modify_prev_results), gated by
vLLM scheduler_config.async_scheduling; maxtext plumbs it as CLI knob
`async_scheduling` (was hardcoded False in the launcher). Flipping it:

- LENDIST out of band FROM STEP 0: truncation 32-49% (band 80-84%),
  osl_mean 1864-2511 (band ~3480). Async generation is ~29% shorter --
  the path is NOT workload-transparent (suspect: per-window RNG handling
  `rng_params_for_sampling = final_rng` differs from the sync path,
  changing the sampling stream). Run stopped at step 7, VOID.
- Even ignoring validity: clean-step rollout 91.7-92.7 s vs 96.7 baseline
  while generating 29% fewer tokens = no per-token win, plus instability
  spikes (rollout 271 s at step 7).
- Verdict: do NOT use async_scheduling with continue_decode until the
  sampling-stream divergence is root-caused. Log: logs/g4r8-async-void.log.
- Collateral fix now in the launcher (297bd6c): the patches configmap keys
  keep `.py` but the roster loop probed extension-less paths -- all 16
  patches silently skipped (fail-open). Loop now accepts both; monitor
  gates on `patches=15` banner lines.

### g4r8b — window 64->128 (sync): COMPLETE, NEUTRAL. Champion stands.

Same champion config, DL_CD=128 (64 -> 32 windows/rollout). Valid run:
jobset Completed, LENDIST on the champion trajectory throughout
(65.5% -> 83.2% / osl 2849 -> 3498). Result (medians, steps>=3, n=16):
global 135.9 (floor 127.0), rollout 99.5, rewards 21.5 (unlucky spike
draw), trainer 18.1. Rollout is ~3 s WORSE than cd=64's 96.7 despite half
the round-trips -- round-trip count is not the binding constraint.
Log: logs/g4r8b-head.log.

### Gap A confirmed: the head runs rollout on 1-2 cores of 44

kubectl top on the live head during g4r8b: rollout-phase usage 0.9-2.1
cores (vs 16.9 cores during the rewards burst -- the 32-proc math-verify
pool parallelizes fine) on a node with 63.7 allocatable cores and a
44-core pod limit. With VLLM_ENABLE_V1_MULTIPROCESSING=0 ("required under
Pathways", launcher line 28), all 128 engine drivers share one Python
process; rollout wall time ~= total host work / 1 core. This single fact
explains the whole Gap-B null series: the 546 ms per-window "wait" is
mostly queueing for the one loop's attention, so neither overlapping it
(g4r8) nor halving round-trips (g4r8b) moves the total.

Rollout-optimization priority, revised:
1. Parallelize per-window host work (process-pool
   _process_continue_decode_outputs; ~30 idle cores available).
2. Revisit VLLM_ENABLE_V1_MULTIPROCESSING=0 under Pathways -- even 4-8
   engine-driver processes with their own proxy clients would multiply
   rollout throughput.
3. Multi-head client sharding (structural).
Kernel/tile work stays pointless until the head parallelizes.
Applies identically to the qwen3 stack (dp16, same uniproc head).

## 🏆 CHAMPION — `g4r4` (2026-08-25): dp32×tp4 + tmbs=8, 128.7 s

Both sweep winners composed on the full 20-step protocol (perf image,
owned-quota lane after the cohort-preemption migration). 20/20 steps.

| Metric (median, steps ≥3, n=17) | g4r4 | g4r2 | g4r1c | g4r0a |
|---|---|---|---|---|
| **Global step** | **128.7 s** (floor 120.1) | 169.2 | 271.2 | 490 |
| Rollout | **99.4 s** | 146.8 | 145.5 | 357 |
| Rewards+advantage | 8.7 s | 6.0 | 12.6 | — |
| Trainer | 17.1 s | 16.4 | 107.9 | ~131 |
| vs baseline | **3.81×** | 2.90× | 1.81× | — |

Per-step globals (step 1 cold 1581 excluded): 275 342 144 147 156 161 125
120 129 131 125 157 181 126 126 131 124 124 128 — steps 2–3 carry the
dp32-mesh decode warmup; steady state 120–131 s. Cross-model: gemma4-e2b
now runs FASTER per step than qwen3-0.6b (133.6 s) on the same stack and
chips — the 4096-vs-8192 completion cap outweighs the 3–4× model cost once
both are fully tuned.

### g4r4 per-step phase table (step 1 = cold; steps 2–3 = dp32-mesh decode warmup)

| phase (seconds) | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 | step 11 | step 12 | step 13 | step 14 | step 15 | step 16 | step 17 | step 18 | step 19 | step 20 | median (4–20) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rollout | 604.6 | 250.2 | 194.9 | 93.0 | 95.3 | 100.6 | 99.4 | 99.4 | 96.2 | 102.9 | 101.1 | 99.6 | 99.9 | 97.2 | 100.4 | 98.5 | 96.6 | 98.6 | 101.1 | 100.5 | **99.4** |
| rewards+advantage | 36.3 | 7.5 | 127.0 | 35.0 | 35.0 | 38.3 | 45.1 | 8.3 | 7.1 | 8.4 | 13.0 | 8.4 | 40.2 | 65.1 | 8.7 | 8.4 | 16.4 | 8.1 | 6.0 | 8.2 | **8.7** |
| train+weight sync | 939.8 | 17.3 | 19.7 | 16.4 | 16.4 | 17.2 | 16.8 | 17.5 | 16.8 | 17.4 | 16.7 | 17.1 | 16.8 | 18.6 | 16.7 | 18.6 | 17.8 | 17.4 | 16.5 | 19.6 | **17.1** |
| GLOBAL STEP | 1580.7 | 274.9 | 341.6 | 144.4 | 146.7 | 156.1 | 161.3 | 125.2 | 120.1 | 128.7 | 130.8 | 125.0 | 156.8 | 180.9 | 125.9 | 125.6 | 130.9 | 124.2 | 123.6 | 128.3 | **128.7** |

Reading: trainer locked at 16.6–18.1 s; rollout locks at 96–104 s from
step 4; rewards mostly 4–9 s with three mild spikes (26–35 s) — the
math_verify fallback path remains the top variance source and the next
median lever (~5 s).

### Packing verdict (`g4r3`, same day) — double negative, rejected

2026-08-24-packing image + tunix sequence packing (MAX_SEQ_TOKEN_PER_TPU=
8192) at dp16×tp8/tmbs=8: global median 222.8 s. (a) Packing does not help
gemma's trainer (18.6 s packed vs 16.4 unpacked — 81%-at-cap uniform
lengths leave nothing to pack; packer overhead eats the remainder).
(b) CORRECTED 2026-08-26: the "packing image rollout is 46 s slower"
reading was OUR patch omission, not the image — image inspection proved
the packing image does NOT bake PRs #3395/#3401/#1904 (unlike our -perf
image, where they are merged), and g4r3 was not mounted with
rpa_blocks/cd_masking/cd_report — so its rollout ran with an inert RPA env
and stock (EOS-early-exit) CD windows. A fair packing-image test requires
the full tianyu-44 11-patch roster. Verdict stands on trainer-side only;
packing remains unattractive for gemma's 81%-truncation regime. tianyu-44 patches that DO
transfer and are now in our tree: phasetime re-anchor (RLEngine),
profiler_fix, gc_sync, packing_knob, packer_fast.

### Infra note — cohort preemption + owned-quota migration

Our `default`-namespace ClusterQueue owns ZERO tpu quota (everything
borrowed from /tpu-shared-cohort); the ubench CI (`cloud-tpu-gu-ubench-*`,
ubench-regression-tests ns) reclaimed capacity and Kueue evicted our runs
mid-flight regardless of priorityClass. Fix: charts now template
`{{ .Release.Namespace }}` and `tools/queue_hunt.sh` detects lanes with
owned quota + our RBAC and migrates the jobs; g4r3/g4r4 completed in
`ubench-regression-tests/multislice-queue` (owned 256 chips; our
priority-500 outranks CI's 100 within the queue).

## 🏆 CHAMPION — `g4r7b` (2026-08-27): packing image + full roster + tp=1, 125.8 s

New stack generation validated as the champion: image `2026-08-24-packing`
(tunix 408ca1d9) + the FULL 16-patch runtime roster (incl. rpa_blocks /
cd_masking / cd_report / reward_parallel, which this image does not bake) +
rollout tp=1×dp128 + tmbs=8 + sequence packing ON + observability
(Perfetto perf-v2, KV gauge, phasetime). 20/20 steps, exit 0; run survived
one mid-run Kueue preemption (attempt 2 rode attempt 1's compile cache:
cold step only 493 s). Contract in band (82.9% trunc / osl 3482).

| Metric (median, steps ≥3, n=17) | g4r7b | g4r4 | g4r5 | g4r0a |
|---|---|---|---|---|
| **Global step** | **125.8 s** (floor 121.6) | 128.7 | 131.3 | 490 |
| Rollout | 96.7 s | 99.4 | 95.4 | 357 |
| Rewards+advantage | 11.7 s | 8.7 | 17.3 | — |
| Trainer | 17.9 s | 17.1 | 16.8 | ~131 |
| vs baseline | **3.90×** | 3.81× | 3.74× | — |

Attribution: tp=1 rollout (≈96 s, confirming g4r5) + newer-stack parity
(the earlier "packing image is slower" was our patch omission — with the
full roster it matches the perf image) + a moderate reward-spike draw.
Packing itself stays ~neutral on gemma (trainer 17.9 packed vs 16.8
unpacked; gc_sync offsets); its value here is being on the newer tunix
that the team's future images build on. Perfetto timeline:
`gs://ubench-logs/alisachen-tpu7x/runs/g4r7b/perf_traces/`.

### g4r7b per-step phase table (step 1 = cold @ warm cache; step 2 warmup)

| phase (seconds) | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 | step 11 | step 12 | step 13 | step 14 | step 15 | step 16 | step 17 | step 18 | step 19 | step 20 | median (4–20) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rollout | 392.9 | 93.0 | 92.0 | 93.3 | 94.2 | 92.9 | 96.3 | 139.1 | 95.3 | 96.9 | 98.1 | 96.7 | 95.8 | 98.4 | 97.4 | 96.7 | 95.4 | 97.0 | 97.3 | 95.0 | **96.7** |
| rewards+advantage | 54.2 | 10.6 | 10.8 | 11.9 | 11.7 | 13.3 | 11.7 | 9.8 | 27.5 | 10.6 | 36.3 | 35.5 | 13.1 | 11.2 | 9.0 | 9.2 | 30.2 | 9.7 | 7.9 | 15.9 | **11.7** |
| train+weight sync | 46.0 | 23.2 | 15.0 | 16.4 | 16.4 | 17.1 | 17.8 | 17.7 | 17.9 | 17.5 | 17.7 | 18.2 | 18.2 | 18.6 | 18.0 | 18.1 | 17.8 | 18.8 | 18.7 | 18.1 | **17.9** |
| GLOBAL STEP | 493.0 | 126.8 | 117.8 | 121.6 | 122.3 | 123.3 | 125.8 | 166.6 | 140.8 | 125.0 | 152.2 | 150.4 | 127.0 | 128.2 | 124.5 | 124.0 | 143.4 | 125.5 | 123.9 | 129.0 | **125.8** |

Reward spikes (28–36 s on 4 steps) remain the gap between the 125.8 median
and the ~122–124 clean-step level — still the top remaining lever.

## tp-sweep completion + GPU-comparable reference (`g4stp1`/`g4r5`, 2026-08-25/26)

Full tensor-parallelism sweep on the champion config (rollout medians):
**tp=8 → 146.8 s · tp=4 → 102.4 s · tp=1 → 95.4 s** — every tp reduction
won; zero-collective decode is optimal for a model this small (Ironwood HBM
bandwidth absorbs the 4× per-device weight traffic; tp=1 also suppresses
the bimodal long-batch rollout mode: 18 of 19 warm rollouts within
90–100 s).

`g4r5` (20 steps, tp=1×dp128, tmbs=8, 20/20 exit 0): **global median
131.3 s, floor 122.7** — rollout 95.4 / trainer 16.8 / rewards 17.3.
Global per step: 129 116 131 127 132 123 276 149 129 131 187 132 128 126
226 257 128 128 177 (cold 1910 excluded). The median trails g4r4's 128.7
purely on reward luck: five math_verify spikes (35–144 s) landed in this
draw vs g4r4's 8.7 s median rewards. Configs are statistically equivalent
on global step; **g4r4 (tp=4) retains the champion title by protocol;
g4r5 (tp=1) is the GPU-comparable reference** — replica-per-accelerator,
the GB200 baseline's topology.

Decisive implication: with clean-reward steps at 122–132 s in BOTH
configs, math_verify spike smoothing (procs 32→64 / verify-input cap /
zero-weight skip) is the last lever to ~118–122 s.

### g4r5 detailed update vs g4r4 (the previous run)

**One workload change and nothing else**: rollout mesh `DL_TP` 4 → **1**
(dp128×tp1 — full model replica per core-device, 2 sequences each; the
GB200 baseline's replica-per-accelerator topology). Byte-identical
otherwise: perf image `head-612fcd2e-gemma4c-perf`, launcher
`gemma4_e2b_r1_launcher.sh`, tmbs=8, RPA `1,8704,1,2176`, CD window 64,
same 6 runtime patches, same workload (256×8 @ 4096+4096, format-weight
rewards, 20 steps), owned-quota lane `ubench-regression-tests`.

Generation-contract check across the whole tp sweep (LENDIST, warm
medians): tp8 82.0% trunc / osl 3447 / 7.06 M tok · tp4 80.4% / 3400 /
6.96 M · tp1 81.1% / 3413 / 6.99 M — within ~1.5%, so the rollout ladder
(146.8 → 102.4 → 95.4 s) is a true per-token speedup. (Contrast: the qwen
ladder's tp4 run tianyu-rl-23b was PARKED because the auto-picked KV page
size changed 128→64 at tp4 and shifted its contract +16% tokens; gemma's
attention geometry lands the heuristic on the same page size at every
mesh. Defensive adoption of their `patch_block_size` + `VLLM_BLOCK_SIZE`
pin is queued for future engine-shape experiments.)

### g4r5 per-step phase table (step 1 = cold; steps 2–3 residual warmup)

| phase (seconds) | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | step 9 | step 10 | step 11 | step 12 | step 13 | step 14 | step 15 | step 16 | step 17 | step 18 | step 19 | step 20 | median (4–20) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rollout | 740.0 | 95.6 | 89.7 | 93.2 | 92.8 | 93.2 | 93.3 | 243.4 | 97.1 | 95.4 | 97.4 | 94.4 | 96.3 | 96.5 | 94.5 | 94.4 | 96.1 | 94.5 | 96.3 | 99.9 | **95.4** |
| rewards+advantage | 59.8 | 16.3 | 9.6 | 21.3 | 17.3 | 22.0 | 11.9 | 16.0 | 34.7 | 16.5 | 16.6 | 74.6 | 18.8 | 14.5 | 15.2 | 114.6 | 144.5 | 16.0 | 15.5 | 60.8 | **17.3** |
| train+weight sync | 1110.3 | 16.8 | 17.0 | 16.7 | 16.8 | 16.6 | 17.5 | 17.0 | 17.1 | 16.8 | 16.7 | 17.8 | 16.7 | 17.3 | 16.8 | 17.0 | 16.7 | 17.3 | 16.3 | 16.1 | **16.8** |
| GLOBAL STEP | 1910.1 | 128.6 | 116.3 | 131.3 | 126.8 | 131.8 | 122.7 | 276.4 | 148.9 | 128.7 | 130.7 | 186.9 | 131.8 | 128.4 | 126.5 | 226.1 | 257.3 | 127.8 | 128.1 | 176.8 | **131.3** |

Reading: rollout is the most stable ever measured — 90–100 s on 18 of 19
warm steps (single 243 s long-batch excursion at step 8); trainer
16.4–18.1 s throughout. The five reward spikes (35/75/115/144/61 s,
clustered late as completion drift feeds math_verify's fallback path) are
the entire gap between the 131.3 s median and the ~122 s clean-step level.


## Rollout mesh sweep (2026-08-25) — dp32×tp4 is a 30% rollout win

Single-variable scout g4stp4b (8 steps, champion config except `DL_TP=4` →
rollout dp32×tp4 instead of dp16×tp8; checkpointing off as scout hygiene):

| | dp32×tp4 (g4stp4b) | dp16×tp8 (g4r2 champion) |
|---|---|---|
| Rollout median (steps≥3) | **102.4 s** (steady 97–103) | 146.8 s |
| Trainer | 17.1 s (unchanged — trainer mesh untouched) | 16.4 s |
| Global median | 156.3 s | 169.2 s |
| Clean-step floor | **125.7 s** | 163.1 s |

### Per-step phase table (g4stp4b, step 1 = cold, all seconds)

| phase | step 1 | step 2 | step 3 | step 4 | step 5 | step 6 | step 7 | step 8 | median (4–8) |
|---|---|---|---|---|---|---|---|---|---|
| rollout | 581.2 | 237.9 | 200.8 | 100.3 | 96.9 | 102.6 | 102.5 | 102.4 | **102.4** |
| rewards+advantage | 33.6 | 27.6 | 5.8 | 9.1 | 92.0 | 37.7 | 35.9 | 6.3 | 35.9 |
| train+weight sync | 853.6 | 16.7 | 16.9 | 17.2 | 17.3 | 16.9 | 17.9 | 17.1 | **17.1** |
| GLOBAL STEP | 1468.3 | 282.1 | 223.6 | 126.7 | 206.2 | 157.2 | 156.3 | 125.7 | 156.3 |

### Where the optimization comes from

The rollout win is a pure decode-mesh effect; nothing else moved (trainer
16.7–17.9 s every step; same kernels, image, patches, workload):

1. **4× fewer TP collective participants per decode iteration.** At tp=8
   every per-token matmul reduce spans 8 devices; at tp=4 it spans 4 — lower
   collective latency on the critical path of all ~4,096 fused decode
   iterations per rollout. Decode is latency-bound per iteration, so
   collective latency multiplies across the whole generation.
2. **2× more independent decode replicas (dp16→dp32)** — 8 sequences per
   replica instead of 16. Smaller per-replica batches shorten the straggler
   tail within each CD window round.
3. **Better weight-read amortization was the tp=8 argument — and it loses.**
   tp=8 halves per-device weight residency vs tp=4, but gemma4-e2b is small
   (~2 GB effective per replica-shard) — HBM capacity was never the
   constraint, so tp=8's collective tax bought nothing.
4. **Cost: a longer warmup** — two full steps (238/201 s) of decode-graph
   compilation at the new mesh vs one at tp8; irrelevant to steady state.

Two caveats recorded: (a) reward spikes (92/38/36 s) now dominate step
variance — the math_verify serial-fallback path, unchanged by the mesh, is
the top remaining variance target; (b) scout is n=5 warm steps — the
combined 20-step run re-measures at full protocol length.

Steps 1–2 (238/201 s rollouts) are extended decode-graph warmup at the new
mesh — excluded. Aysu's serving-side tp=4 evidence transfers to RL decode.
Remaining variance: math_verify spikes (92/38/36 s on 3 of 5 warm steps).
Scout attempt 1 (g4stp4) was killed twice by the tensorboardX/orbax GCS-412
write race at the step-0 checkpoint flush; disabling checkpointing for scouts
is the workaround. Next: combined run dp32×tp4 + tmbs=8 + sequence packing
(g4r3 validating packing on the 2026-08-24-packing image now).

## Trainer microbatch sweep (2026-08-22) — Fix #21 wall demolished

Single-variable tmbs scouts on the R0 stack config (8 steps each, medians
steps>=3; rollout is tmbs-independent, so trainer segment is the read):

| tmbs | trainer segment | scout global median | run |
|---|---|---|---|
| 1 | 108.1 s | 271.2 s (20-step champion) | g4r1c |
| 2 | 55.5 s | 234.3 s | g4s2b |
| 4 | 29.7 s | 199.0 s | g4s4b |
| **8** | **15.8 s** | **182.9 s** | g4s8 |

Perfect halving per doubling through tmbs=8. The wall (probed 2026-08-24,
scouts g4s16/g4s32): **tmbs=16 OOMs by 0.78 G** (95.52 G needed vs 94.74 G
per-device HBM) and tmbs=32 needs 120.64 G — so **tmbs=8 is optimal and
final** for this trainer path. Confirmed twice: `remat_policy=full`
(g4s16r, 2026-08-25) produced the byte-identical 95.52 G OOM — the dominant
temporaries are the loss-layer 262k-vocab logprob tensors, which
rematerialization cannot reduce. Fix #21's "tmbs=1 mandatory" was
config-stale (the old config needed 130 G at tmbs=4; the current stack fits
tmbs=8 and made tmbs=32 cost what tmbs=4 used to). Matching qwen's tmbs=32
microbatching requires the structural fix: fused/chunked logprob
computation that never materializes the (tmbs, seq, 262144) tensor —
worth ~8 s more; upstream candidate. Also swept and REJECTED:
CD window 128 + EOS-interval 8 (g4s1: 441-563 s — gemma's rollout is not
dispatch-bound; window 64 suffices). Champion rerun at tmbs=8 (g4r2, 20
steps) in flight for the official number. Remaining variance: rollout
long-batch mode (~140 vs ~230 s alternation) and math_verify fallback
spikes (one 501 s step in g4s8).

Scout-failure notes: g4s2 attempt 1 died of the chart-c RM 16G OOM (Fix #24
applied to chart-c too); g4s4 attempt 1 died of a vpod webhook race then a
tensorboardX GCS 412 collision between jobset attempts (fresh release name
is the workaround).

## Tuning probe ladder (in progress)

- **RPA block-size forwarding proven live** (#3395 path): probe g4p2's tuple
  was *rejected by the kernel's validator* (`bkv_sz % bkv_csz != 0`) — which
  only fires when the override reaches the kernel; g4p3 with `1,8704,1,2176`
  ran 2 clean decode-heavy steps. Constraint recorded: at kv=8704 the valid
  chunk ladder is {4352, 2176, 1088}. Phased-profile kernel-name verification
  pending (trace didn't flush in a 2-step probe — capture fix queued).
- Remaining rungs: no-env control at probe shape, bkv_csz 4352 vs 2176 vs
  default, `USE_BATCHED_RPA_KERNEL=1`, KV `block-size=256` (both from aysu's
  gemma4 serving config), dp32×tp4 vs dp16×tp8 (aysu evidence favors tp=4),
  tmbs=2 trainer scout, zero-weight-reward skip (~30 s/step cold observed),
  then the **decode-heavy stack run `g4r1`** vs g4r0a's 490 s median.

## Cost of measurement honesty (found and fixed along the way)

Cross-tenant Pathways head port collision (Fix #22), helm `workload.envs`
array replacement wiping ARTIFACT_DIR, launcher port overrides vs port-shifted
charts, kernel tuple divisibility, phased-profiler gs://-only requirement —
all recorded in RUNBOOK/repo for the next campaign.

---

## Tuning campaign — progress, per-change attribution, and final numbers

### The run ladder (chronological; one labeled change per rung)

| Run | Change vs previous | Regime | Result (warm) | Verdict |
|---|---|---|---|---|
| t6g10 / t6g13 | first working gemma4 recipe (MaxText route, tmbs=1) | smoke 2x2x1 | 2/2 steps pass | recipe validated |
| t6gn8 | native `Gemma4ForCausalLM` route (weight-mapping v3) | smoke 2x2x1 | 2/2 steps pass, 0 unmapped | second route validated |
| **g4r0a** | benchmark scale-up, NO optimizations | decode-heavy 4x4x4 | **490 s median cycle** (rollout 357 s, train 131 s) | baseline-of-record |
| g4r0p | + upstream PRs #3395/#3401/#1904 baked in (default config) | decode-heavy 4x4x4 | ≈ baseline (312/324 s aligned rollouts, train 131 s identical) | **+0% — PRs are activation-gated** |
| g4dls3 | + full tianyu stack ACTIVATED (5 patches, CD-64 window, sharded logprobs tp≥2, async off, XLA flags, format-only rewards) | decode-light smoke 2x2x1 | all banners APPLIED; regime + mechanics proven | stack validated on gemma |
| **g4dl** | stack at benchmark scale, frozen DL invariants | decode-light 4x4x4 | **14.1 s median / 12.4 s floor**, 4,635 tok/s | **champion** |
| g4p2/g4p3 | RPA block-tuple probes (`1,8704,1,{2048→2176}`) | decode-heavy probe 2x2x1 | forwarding proven live (kernel validator fired, then clean run) | rung open — timing pending |

### Gain attribution — what each optimization commit actually bought

| Optimization | Evidence | Contribution |
|---|---|---|
| Upstream PRs merged into the image (#3395 RPA forwarding, #3401 CD window correctness, #1904 reward pool) — **unactivated** | g4r0p aligned A/B vs g4r0a | **0%** alone. The code is necessary-but-not-sufficient; every gain below runs *through* it. |
| **Continue-decode fused window (depth 64)** = #3374 port (`patch_cd_proper`) + #3401 correctness + `enable_continue_decode` | Prior TPU baseline's own profiling: rollout ~78% dispatch-idle; g4dl generation = 8.8 s for 65,536 tokens where the baseline stack produced ~11k tokens in ~90+ s | **The dominant contributor** — eliminates per-token host round-trips; order-of-magnitude effect in the dispatch-bound regime |
| Sharded logprobs (`patch_sharded_sampling`, tp=8) | active in g4dl (sampling half fell back by design at top_k=−1); removes the 262k-vocab replicated log_softmax/top-k per iteration | Contributor within the 8.8 s generation phase; isolated share not yet measured (needs SHARDED_SAMPLING=0 ablation) |
| Format-only rewards + warm reward path | phasetime: 30.4 s cold → **0.73 s warm** (vs tianyu's 29.8 s serial measurement on qwen) | Rewards are a non-issue warm in DL; the ~30 s cold hit is one-time. Zero-weight-fn skip remains a decode-heavy rung |
| `async_scheduling=False`, XLA proxy flags (scoped_vmem 65536), compile cache, `maxRestarts=5` | adopted wholesale from tianyu/aysu configs; not individually ablated | Bundled into the champion config; flag ablation is a pending rung |
| RPA block-size tuple | g4p3: forwarding live, tuple constraint mapped (kv 8704 → chunks {4352, 2176, 1088}) | No timing verdict yet — irrelevant in DL (kernel is a sliver at 64-token decode), matters for the decode-heavy rung |

**Net, like-for-like:** within the decode-light regime on TPU v7x 64-chip
hardware, the activated stack moved the step time from the pre-stack baseline's
**109 s median → 14.1 s (7.7×)**, and stands **8.7× ahead of the GB200 122.8 s**
reference arm. (g4r0a's 490 s is a *different regime* — decode-heavy math RL —
and is the pending comparator for `g4r1`, not a numerator for these ratios.)

### Final performance summary (as of 2026-08-18)

| Regime | Config | Step time (median, warm) | Throughput |
|---|---|---|---|
| **Decode-light (GB200-matched)** | full stack, `g4dl` | **14.1 s** (floor 12.4) | **4,635 tok/s** |
| Decode-heavy (real math RL) | unoptimized baseline, `g4r0a` | 490 s | ~2048×~3500-token completions / cycle |
| Decode-heavy + stack (`g4r1`) | — | *pending* | expected: rollout-bound 357 s attacked by CD window + sharded sampling (full path at top_k=50) + tuned RPA |

### Open rungs (priority order)

1. `g4r1` — decode-heavy + full stack vs 490 s baseline (the real-workload payoff number)
2. RPA chunk timing probe with working trace capture (needs >2-step probe for the phased profiler to flush) + control
3. SHARDED_SAMPLING=0 ablation (isolate its share)
4. `USE_BATCHED_RPA_KERNEL=1` and KV `block-size=256` (aysu's gemma4 serving evidence)
5. dp32×tp4 vs dp16×tp8 (aysu evidence favors tp=4 for 1-KV-head gemma)
6. tmbs=2 trainer scout (halves the 131 s segment's 256 grad-acc microsteps if it fits)
7. Zero-weight-reward skip; XLA flag ablation

## Validity audit of the champion run (performed post-hoc, 2026-08-18)

Prompted by "the perf seems too good." Verdict: **valid**. Key evidence, in
strength order:

1. **The model learned during the run** — `rewards/train/match_format_exactly`
   rises 0.041 → 0.100 (100% match rate) by step 6, with grad norms
   287 → 336 → 24 → 8 → 3 → 1 → ~0 as the policy converges. A rising reward
   curve transitively verifies every link: coherent generations, correct
   logprobs, real gradients, and a weight push landing in the sampler every
   step. Garbage or stale weights cannot produce a monotone learning curve.
2. Invariants behaviorally confirmed: effective `rl.num_generations=8` /
   `rl.grpo_beta=0.0` (two alarming top-level config echoes — `num_generations:
   2`, `grpo_beta: 0.08` — are unused legacy fields; `actor/train/kl=0.0` on
   every step proves β=0 in effect). `completions/mean_length` → 64.0 matches
   LENDIST independently.
3. `loss=0.000000` is expected for on-policy GRPO (group-normalized advantages
   ⇒ E[loss]≈0), not an anomaly; gradients are nonzero.
4. Timer honesty: `async_scheduling=False`, and LENDIST reads host-side token
   arrays inside the measured window — generation demonstrably completed
   within the reported 8.8 s.
5. Physical plausibility: 7.5k tok/s over 64 chips ≈ 117 tok/s/chip for a 2B
   model — still overhead-bound, far below memory-bandwidth limits; the
   references were simply more overhead-bound (78% dispatch-idle, their own
   profiling).

Caveat recorded: gradients ≈0 after step ~7 (format task solved) — compute per
step is unchanged, so timing validity holds, but the decode-light benchmark
measures system throughput, not sustained learning (equally true of the GB200
spec it mirrors).

# ═══ G4R19 ROOFLINE & BOUND ANALYSIS — tp1×dp128 full-slice (2026-09-01) ═══

## Measured (g4r19, 8 steps, exit 0)
steps 1-7 global: 367, 176, 153, 155, 149, 427, 350 → median 175.6, clean band 148-155.
Clean-step split: rollout ~125-130, rewards 8.5-17.5 (spike 128), trainer+rest ~15.
LENDIST red flag: truncation RISES 64%→78% over 8 steps (osl_mean 2745→3304) — missing gemma
chat stop ids (106/50); training-on-truncated feedback loop. Fixed in g4r20.

## Roofline (per device: ~2.3 PFLOP/s bf16, ~7.37 TB/s, ridge ~313; 128 devices)
- Decode: 2048×osl~3200 ≈ 6.55M tokens/step. Per engine-iteration bytes ≈ active weights ~5.2GB
  + KV ~0.7GB ≈ 6GB → 0.81 ms/iter at the memory roof → ideal device decode ≈ 2.6 s/rollout.
  MEASURED 125-130 s → **device busy ~2-3%; ~97% host** (head loop + Pathways RPC + collect).
- Window arithmetic: ~50 windows(64-deep)/rollout; ~2.5 s/window wall vs ~52 ms device.
  Host cost/window scales with dp: ~0.78 s @dp64 → ~1.0 s @dp32(tp2) → ~2.4 s @dp128.
- Trainer: ideal ~0.4-1 s (1.06e17 FLOPs / 128 dev) vs measured ~15 s (TMBS=4 micro-step count,
  small-model MFU).
- Rewards: pure host python.
**BOUND: HOST at every phase. Roofline gap ≈ 30-50× (ideal step ~4-6 s vs 148-175 measured).**
Consistent with all traces (g4r7b/g4r8 xplane+Perfetto device_get 546ms/window; tp2/tp4 host
97-98%; tianyu-rl-148 block_until_ready-dominated). No new trace capture needed to identify the
bound; per-increment phasetime is the discriminating instrument.

## Same-mesh gap to tianyu g4-003 (their 51 s vs our 175 median)
| Phase | ours | theirs | their levers (ordered) |
|---|---|---|---|
| rollout | 125-130 | ~40 | CD window 512 (8× fewer round-trips), async detok/logprobs, stop-ids
(shorter osl), rr routing, prefix off, block pin 128, XLA proxy ladder |
| rewards | 8.5-17.5 +spikes | 0.3 | async_reward during rollout + fast_extract + format_guard |
| trainer+rest | ~15-23 | ~11 | TMBS=8 (flash-enabled), same fsdp128 |

## INCREMENTAL LADDER (all 8-step, tp1×dp128 full-slice base = g4r19)
E1 g4r20 stop-ids [106,50] (RUNNING) — truncation/correctness + osl↓ → rollout↓
E2 g4r21 +DL_CD=512 — biggest single rollout lever (cd_proper/masking/report already in roster)
E3 g4r22 +prefix-off +rr-routing (composed re-test at dp128)
E4 g4r23 +async detok/logprobs/reward (port 3 patches) — rewards → ~1 s
E5 g4r24 +TMBS=8 — trainer micro-step halving
E6 g4r25 +XLA proxy-flag ladder (chart-level change)
E7 (flagged, semantic): gemma-3-style textual prompt — changes the RL problem framing; user call.
Gate: keep an increment iff clean-step median improves; revert otherwise. Rolling best carried forward.


# ═══ G4R30: TIANYU RECIPE REPRODUCED VERBATIM (2026-09-01 ~21:30Z) ═══
Cloned g4-003 configmaps byte-exact (same image build as ours: tag 2026-08-24-packing == sha f7c957d0),
jobset reconstructed, run as g4r30 on our namespace/queue. 10/10 steps, EXIT 0.

steps 1-9 global: 55.8, 49.8, 52.3, 50.1, 50.4, 62.4, 49.8, 50.3, 48.9 -> WARM MEDIAN ~50.2 s
rollout ~38.4-41 (one 51.6), rewards 0.21-0.56 s, cold step 205 s (their compile cache).
rewards/train/check_numbers 0.011-0.022 == BYTE-IDENTICAL to their g4-003 series (deterministic repro).

=> The ~50 s / step target stack now runs under our control with full logs/TB. Our-launcher best:
111-125 (tp2 half-slice). Remaining gap to attribute: cd512+cond-fix, async host work, prompt,
TMBS8, XLA ladder — g4r21c (cd512-fixed + prompt + stop-ids on OUR launcher) running now.


# ═══ G4R21C VERDICT + ATTRIBUTION PIVOT (2026-09-01 23:30Z) ═══
g4r21c (our launcher: cd512 + FIXED cond + gemma-3 prompt + stop-ids): step 0 now fast and reliable
(370 s, rollout 224-234 s; prompt reproduces tianyu's generation contract exactly: 81.8%/3462) BUT the
synchronized worker crash STILL fires ~9-10 min into step 1 (5 restarts, dumps at 23:03:16).
=> cond all-reduce was a real defect, NOT the whole killer. Remaining launcher deltas vs the working
recipe: body-reduce patch variant (our cd_noeos vs their cd_noglobal), unpinned VLLM block size,
max_num_batched_tokens 8448 vs 33280, and the 30-flag XLA proxy ladder (collective-scheduling flags
possibly load-bearing for 512-deep fused programs).
PIVOT: attribute from the WORKING side (vary g4r30) instead of patching toward it.
- g4r31 (launched): g4r30 minus all 27 XLA proxy flags -> tests whether the ladder is required for
  deep-window stability AND its perf contribution in one run.
- next candidates: g4r32 = g4r30 @ window 64 (H1 sizing on the working stack); g4r33 = g4r30 with
  canonical prompt (isolates prompt's perf-vs-correctness split).


# ═══ 🏆 G4R32: NEW CHAMPION — TEXTUAL-STOP ON THE PROVEN BASE (2026-09-02 01:49Z) ═══
g4r32 = g4r30 recipe + patch_textual_stop (TEXTUAL_STOP=1) + reward_dump. 10/10 steps, EXIT 0.

warm steps 1-9 global: 46.5, 39.5, 41.7, 40.3, 42.0, 56.2, 43.4, 204.8, 58.9
-> CLEAN BAND 39.5-43.4, MEDIAN 43.4, BEST 39.5  (g4r30 reference: 48.9-55.8, median 50.2)
rollout: 33-36 s clean (vs 38-41)  |  cold step: 201 s
OSL: 3459-3600 -> 1748-2102 (-45%); truncation 81-85% -> 37-47%
REWARDS IMPROVED: check_numbers 0.019-0.0283 (vs reference 0.011-0.0215) -- ending at the real
answer instead of self-dialog RAISES correct-answer density. Task healthier AND faster.

Net vs our campaign start: 125.8 -> 43.4 median (2.9x); vs roofline (~4-6 s): ~9x residual (H8).
Remaining tails: one reward spike 156.6 s (step 8: async-reward fallback to sync on a hard batch);
rollout wobble 48-53 s on 2 steps. Next increments: window-boundary overshoot trim (cd 256 probe),
reward-fallback hardening, include-stop-str contract alignment.
ATTRIBUTION NOTE: the textual-stop is OUR novel lever (not in tianyu's stack) -- first result
surpassing the reference.


# ═══ REWARD-CONFIG AUDIT (2026-09-02) ═══
Summer's GPU reward (maxtext_math_reward.py): correct=+1.0 (exact/whitespace/math-verify), format=+0.1.
rl.yml defaults MATCH exactly (reward_exact_answer=1.0, white_space=1.0, format=0.1, penalties 0.0);
math-verify success awards reward_exact_answer (math_verify_pool.py:246).
FINDING: OUR helm launcher passed reward_exact_answer=0.0 + white_space=0.0 since bring-up ->
in ALL our-launcher runs (g4r7b..g4r21d) CORRECT ANSWERS SCORED 0.0 (format-only RL signal).
g4r19's zero check_numbers had TWO stacked causes: broken prompt AND zeroed reward. FIXED (overrides
removed; defaults now apply). The g4r30/g4r32 lineage never had the bug (tianyu launcher does not
override) -> g4r32's rewards are directly comparable to Summer's GPU scale; its check_numbers
0.019-0.028 = true ~2-3% accuracy at step 0, temp 0.8.
