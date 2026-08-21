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

### Side result — qwen3 ladder point run for tianyu (run-17 → `tianyu-rl-17a`)

Teammate's RPA decode-tile sweep point 2 (`1,1024,1,1024`), their manifest
verbatim (renamed 17a; eviction guard added): 20/20 steps, exit 0, all bands
green. **Official read: 133.6 s median global (steps ≥3) vs parent 15b's
126.1 s — the 1024 fetch/compute tile is a ~6% regression, entirely
rollout-side (92.1→105.4 s), train unchanged (24.8→25.1 s).** Matches the
pre-registered DMA-bandwidth risk; ladder keeps `1,16384,1,4096`.
Artifacts: `gs://ubench-logs/tianyu-rl-17a/`.

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
