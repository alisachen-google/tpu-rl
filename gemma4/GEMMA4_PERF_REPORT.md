# Gemma4-E2B RL Performance Report — TPU v7x (Ironwood) 4x4x4

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

## Trainer microbatch sweep (2026-08-22) — Fix #21 wall demolished

Single-variable tmbs scouts on the R0 stack config (8 steps each, medians
steps>=3; rollout is tmbs-independent, so trainer segment is the read):

| tmbs | trainer segment | scout global median | run |
|---|---|---|---|
| 1 | 108.1 s | 271.2 s (20-step champion) | g4r1c |
| 2 | 55.5 s | 234.3 s | g4s2b |
| 4 | 29.7 s | 199.0 s | g4s4b |
| **8** | **15.8 s** | **182.9 s** | g4s8 |

Perfect halving per doubling, zero trainer OOMs — Fix #21's "tmbs=1
mandatory" was config-stale (measured pre-stack; the pretrain recipe's
per_device_batch_size=2 @ 8192 was the tell). Also swept and REJECTED:
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
