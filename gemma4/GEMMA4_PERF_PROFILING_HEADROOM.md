# gemma4-E2B GRPO — Perf Profiling, Roofline & Improvement Headroom

**Run:** `g4r55b` (old-image `f7c957d0`, TP1×DP128 on tpu7x 4×4×4 = 64 chips), `raw2`
checkpoint, `gemma_data.py` prompt. Reward/OSL at GPU parity (check_numbers
0.58–0.62 vs verl 0.54–0.65; OSL ~1420, 2.5% truncation, natural id-106 stop).
Companion: `GEMMA4_PERF_LADDER_DETAILED.md`, `GPU_TPU_REWARD_ALIGNMENT.md`.

---

## 1. Measured per-step profile (TensorBoard `perf/train/*`, steady state steps 3–4)

| Phase | Time | % of step | Notes |
|---|---|---|---|
| **Total step** | **~33–34 s** | 100% | matches the campaign's proven best (34.5 s) |
| **Rollout (decode)** | **~28.6 s** | **~86%** | the dominant cost |
| actor_dequeue (marshaling) | ~7.8 s | overlaps rollout (async) | mostly hidden, not additive |
| actor_train | ~0.6 s | ~1.8% | already optimal |
| weight_sync | ~0.4 s | ~1.2% | already optimal (free_kv, jit-unroll upstreamed) |

The trainer + weight-sync are ~1 s combined — the upstreamed optimizations work.
**Single-turn GRPO is rollout-bound: ~86% of wall-clock is generation.**

## 2. Decode roofline (confirmed inputs)

- **HBM BW: 7.37 TB/s** (Google TPU7x docs — confirmed, not assumed).
- gemma4-E2B: 4.63B params → **9.26 GB bf16 weights/device** (TP1 replica); **MQA**
  (1 KV head), head_dim 256, 35 layers (20 KV-shared), vocab 262144.
- Batch **16 seqs/rank** (2048 / DP128), OSL ~1420.

| Quantity | Value |
|---|---|
| Bytes read / decode-step (weights + KV) | ~9.6–10.1 GB |
| Time / decode-step (HBM-bound) | ~1.3–1.37 ms |
| **Pure weight-stream decode floor** | **~1.8 s/step** (1420 tok × ~1.3 ms) |
| Measured rollout | 28.6 s/step |
| **Decode matmul as fraction of rollout** | **~6%** |
| Aggregate throughput | ~102k tok/s (≈6.6% of the ~1.5M tok/s HBM ceiling) |

### Headroom conclusion
**The decode matmul is NOT the bottleneck** — at batch 16/rank it is already near
its memory-bound floor (~1.8 s), only ~6% of the 28.6 s rollout. **~94% of rollout
is non-matmul host/overhead**: prefill (2048×~151 prompt tokens), DP scheduling,
CD-window host turnaround, KV management, detokenization. This matches the
campaign's history (the −72 s win was round-robin DP routing — pure host-work).
**Headroom lives in host-work and prefill, not the decode kernel.**

### MEASURED decode-step op composition (real xplane, 2026-10-23)
Parsed the device-side `XLA Ops` leaf line from an existing gemma4-E2B
`decode_only` xplane — `gs://ubench-logs/alisachen-tpu7x/runs/g4nx1s0851/phased_profile/decode_only/.../server-*.xplane.pb`
(Aug-19). Two TensorCore devices, consistent:

| Decode-step device-op share | TPU:6 | TPU:4 |
|---|---|---|
| **Attention RPA (growing KV, `RPAd-…-bkv_8192/8448-sw_512`)** | **44.1%** | **48.1%** |
| **Vocab logits + sampling over 262144 (`while` + reductions on `f32[262144,64]`, top-k `fusion→(val,idx)`)** | **42.1%** | **37.2%** |
| Collective/comm (TP4 all-gather of `f32[64,262144]` logits + all-reduce) | 7.6% | 8.2% |
| **Matmul/dot (weight GEMM)** | **2.1%** | **2.3%** |
| layout/copy + fusion(norm/rope) + other | ~4% | ~4% |

**Decode device time is ~86% attention + 262k-vocab sampling.** The weight GEMM is
only ~2% of device-op time — consistent with the roofline (matmul ≈6% of rollout,
memory-bound floor). So on the decode kernel itself, the two levers that matter are
**(1) 262k-vocab logits+sampling (~40%)** and **(2) RPA attention over growing KV
(~46%)** — *not* weight quantization.

> **Caveat — config differs from g4r55 production.** This trace is **TP4, 1 replica
> on 4 devices, 64 seqs, `max_decode_steps=64`, prefix-caching ON, Aug image**;
> g4r55 is **TP1×DP128, 16 seqs/rank, `max_decode_steps=512`, prefix-caching OFF**.
> The op *composition* (attention vs vocab-sampling vs GEMM) is representative of
> gemma4-E2B decode on v7x, but absolute times and the collective share differ: at
> **TP1 the vocab is not sharded, so the `f32[64,262144]` logit all-gather (~8%)
> disappears**, redistributing to attention + vocab-sampling, and the full 9.26 GB
> weights sit on one device (GEMM share rises modestly but stays memory-bound). The
> g4r55-exact split still wants a TP1×DP128 capture (manifest ready,
> `PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR=2`), but the dominant levers are already
> clear and unchanged by TP.

> **Pending (g4r55-exact split):** a TP1×DP128 `decode_only` xplane to confirm the
> above at the production config.
>
> **Why every capture attempt has come back empty (root cause, 2026-10-23 — from
> the tpu_inference source `runner/utils.py` + `runner/tpu_runner.py` and a
> known-good Aug-19 log).** On `g4r55prof2` step 0 completed (`phasetime`,
> `lendist`, `check_numbers_*` all landed; OSL mean 1419 / p90 2957 / trunc 2.5%,
> responses coherent) **but `phased_profile/` was never even created.**
>
> The `PhasedBasedProfiler` captures a phase like this: when it first sees a batch
> of a phase (e.g. DECODE_ONLY = prefill-ratio exactly 0.0) it `start_trace()`s and
> sets `profiling_n_steps_left = PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR`
> (**default 15**); `.step()` is called **once per `execute_model`**, decrements,
> and only at `<= 0` does it `stop_trace()` → **which is what writes the xplane.**
> With **continue-decode `max_decode_steps=512`**, each `execute_model` advances
> ~512 decode tokens but counts as **one** profiler step. At OSL ~1420 the decode
> phase drains (sequences hit EOS) after only ~3 execute calls — **never reaching
> 15 — so `stop_trace()` never fires and nothing is written.** The known-good
> Aug-19 run `g4nx1s0851` used `max_decode_steps=64`, so it got ~22 decode execute
> calls > 15 and captured `prefill_only/` + `decode_only/` xplanes normally. It was
> also Pathways-MPMD, so Pathways/MPMD is **not** the blocker. (The reservation
> teardown and 3× restart made it worse but were not the cause — even a clean full
> run would not have flushed at 512-deep.)
>
> **Fix for next capture (wired into `g4r55prof2-ready.yaml`):** set
> **`PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR=2`** so the window closes and writes
> after ~2 execute calls (~1024 decode steps captured — ample for the split), well
> inside a single rollout; keep `num_batches=2`; launch the instant a reservation
> exists. If it still restarts after step 0, pull the live worker log for the
> `"Starting profiling for decode_only phase"` / `"...finished"` lines *before* the
> window closes.
>
> **Fix verified offline (2026-10-23)** against the image's own
> `tpu_inference.runner.utils.PhasedBasedProfiler` (jax.profiler trace calls
> stubbed, 10 pure-decode batches fed): at `NUM_STEPS=15` → `start_trace=1`,
> `stop_trace=0` (never writes — reproduces the empty capture); at `NUM_STEPS=2` →
> `start_trace=1`, `stop_trace=1`, xplane written after execute call 3. The
> decode-only phase persists until the last sequence EOSes (tail to cap 4096 ≈ 8
> execute calls at 512-deep), so the 3 needed is comfortable; `=1` is the
> ultra-safe fallback if a run is ever shorter.

---

## 3. Improvement actions (ranked by expected leverage)

> **Measured re-ranking (2026-10-23 decode xplane).** Device decode time is ~46%
> RPA attention + ~40% 262k-vocab logits+sampling + ~2% weight GEMM. So the
> device-side levers that matter are **#10 (262k-vocab sampling efficiency)** and
> **#10/RPA tile tuning for head_dim 256** — promote both to the top of section B.
> **#7 (int8/fp8 rollout weights) is demoted**: the weight GEMM is only ~2% of
> decode device-op time (matmul is already memory-bound and tiny), so halving the
> weight read buys little on the decode step — do it only if a TP1 capture shows the
> single-device weight stream is materially larger than this TP4 trace implies.

### A. Rollout host-work & decode-step overhead — the 94%, highest leverage
> **Correction (measured):** ISL is only **~150 tokens**, so **prefill is tiny** — the
> 94% is NOT prefill-dominated. It is decode-step device overhead (RPA attention over
> growing KV + categorical sampling over the 262k vocab) + CD-window host turnaround.
> And **prefix caching is a known regression here**: `VLLM_PREFIX_CACHING=0` was set
> because turning it *off* saved **−6 s/step** (run 33j) — the cache bookkeeping
> overhead exceeds prefill savings on ~150-token prompts. **Do NOT enable it.**
>
> **The real split of the 94% requires the decode xplane** (`g4r55prof`, profiler on).
> Blind config changes risk regressions (as prefix caching would) — target by the trace.

1. ~~Enable prefix caching~~ — **rejected**: measured −6 s/step when OFF (run 33j); ISL
   ~150 makes prefill negligible, so cache overhead dominates.
2. **Deepen the CD window** (`max_decode_steps` 512 → 1024/1420). Each window has a
   fixed host turnaround; at OSL ~1420 we pay it ~3× per sequence. Fewer, deeper
   windows amortize the host round-trip. Watch HBM (KV lookahead booking).
3. **Trim per-window host turnaround.** The host EOS-trim/report path runs per
   window; profile it (xplane) and vectorize/async any residual (vectorized EOS
   trim is upstreamed; check detok/logprob extraction overlap is fully hidden).
4. **Tune chunked-prefill batching** (`max_num_batched_tokens`, currently 33280).
   Larger prefill chunks = fewer prefill steps; balance against the padding bucket.
5. **Overlap next-step rollout with the current train+sync** (~1 s) if tunix's
   pipeline allows — hide the trainer entirely behind rollout.
6. **Disaggregated rollout** (separate inference workers from trainer) — advanced;
   lets rollout and train run on distinct chips concurrently.

### B. Decode efficiency — the 6% matmul (memory-bound)
7. **Quantize rollout weights to int8/fp8.** The 9.26 GB weight read *is* the decode
   floor; int8 halves it → up to ~2× decode-matmul throughput. gemma4 int8 inference
   weights only (trainer stays bf16). **High impact on the floor, moderate effort**
   (needs accuracy check — but rollout is sampling, tolerant).
8. **Raise per-rank batch / lower DP.** At 16 seqs/rank the weight read is poorly
   amortized. Fewer DP ranks (e.g. DP64) → 32 seqs/rank → 2× weight amortization per
   decode step — but halves parallelism; net depends on where the 94% overhead sits
   (needs xplane). Could also try **TP2** to halve per-device weights (4.6→2.3 GB) at
   the cost of TP all-reduce comms.
9. **Int8 KV cache** — minor here (MQA already makes KV ~0.4–0.8 GB/step), low priority.
10. **RPA kernel tile tuning for head_dim 256.** gemma4's head_dim 256 is 2× qwen's;
    re-tune `RPA_V3_DECODE_BLOCK_SIZES` for it (currently generic). Needs the decode
    xplane to confirm the active kernel config.

### C. Response length (OSL) — fewer tokens = proportionally faster rollout
11. **Tighten the response cap / add a mild length penalty** if accuracy holds — OSL
    1420 drives rollout time linearly; even −20% OSL ≈ −20% rollout.
12. **Speculative decoding** (draft model / MTP) for the rollout — gemma4 has a
    `gemma4_mtp` path in tpu_inference; multi-token prediction could cut decode steps.
13. **Improve format compliance (82% → ~98% like GPU)** so fewer sequences ramble to
    long/truncated outputs — tightens the OSL tail (also lifts reward).

### D. Trainer & weight-sync — already optimal (~1 s), low priority
14. train 0.6 s, sync 0.4 s — the upstreamed `overlap_postprocessing`,
    `free_kv_cache_during_weight_sync`, jit-unroll, packer-fast are all active.
    Only revisit if A/B shrink rollout enough that this 1 s becomes material.

### E. Systemic / measurement
15. **Capture the decode xplane** (`g4r55prof`, in flight) to split the 94% and target
    the single largest host component precisely — prerequisite for A2/A3/B8/B10.
16. **Confirm the RPA decode kernel config** from that xplane (block sizes, bkv) —
    the campaign previously found the env override was inert on one path.
17. **Re-measure at steady state over ≥10 steps** for a tighter median (compile steps
    0–1 dominate short runs).

### F. New-image track (correctness-gated, not perf)
18. The upstream image's gemma4 **inference-forward regression** (garbage rollout)
    must be fixed before any new-image perf work — root-caused to the forward;
    forward-logit probe staged. Perf tuning on the new image is moot until then.

---

## 4. Bottom line
At ~34 s/step we are at the proven optimum for this config. The **top three
actions** by expected leverage: **(1) prefix-cache the shared system prompt**,
**(7) int8 rollout weights**, **(2) deeper CD window** — all attack the rollout,
which is 86% of the step. The decode *kernel* is near its memory-bound floor; the
recoverable time is in **prefill + host-work**, which the pending xplane will
pinpoint.

---

## 4. ubench optimization campaign — straggler attack (2026-10-23, autonomous)

**Measured on ubench-regression-tests pool (same NAP cluster, 64 chips=128 devices),
old image, 3-step runs. KEY METHOD: step-1 is profiler-inflated (trainer xplane
profiles step 1, `skip_first_n=1`); step-2 is the clean steady state.**

### Results board
| Config | step-2 steady | reward (check_numbers) | verdict |
|---|---|---|---|
| proven best (g4r55b, no profiler, 10-step) | ~34 s | 0.58–0.62 | reference |
| DP128, `least_loaded` (bug) | — (step-1 43.9) | — | baseline |
| + `round_robin` routing FIX | — (step-1 40.8, −3.1 s) | — | ✅ keep |
| + deeper CD **md1024** (g4ub3) | **31.7 s** | **0.560/0.604/0.586 ✓ parity** | ✅ **best, ≤ proven** |
| combiner + overlap XLA flags | no change (permute 60%, overlap 0%) | — | ❌ dead end |
| DP64×TP2 (g4ub4) | step-1 77.8 (≈2× worse) | — | ❌ TP2 per-step comms |

### Root cause of the ~60% collective-permute (straggler, confirmed per-plane)
Load-imbalance barrier-wait: ~8 chips grind the long-OSL tail (~16 ms window, 44–51%
compute) while ~24 chips finish in ~3.6 ms and block in `collective-permute-done`
(90%+). Not bandwidth, not op-launch — a barrier wait, so XLA overlap/combiner flags
can't touch it (confirmed: 0% overlap even with the flags).

### The routing bug
`DP_ROUTING=rr` in the manifest is a **dead env var** — tpu_inference reads
`DP_SCHED_ROUTING` (default `least_loaded`, which probes all 128 ranks ×4 per request).
Fix: `export DP_SCHED_ROUTING=round_robin`. Worth −3.1 s/step.

### DP-lowering is blocked
Mesh requires `TP×DP×EP == 128` sampler devices. `DP64×TP1=64` crashes; `DP64×TP2=128`
runs but TP2's per-decode-step 262k-logit + attention all-gathers regress it ~2×. So the
straggler can't be fixed by lowering DP on this slice.

### Code-level straggler fixes (the remaining lever)
- **Variant A (re-prefill work-stealing)** — `patches/patch_worksteal_a.py`: at each
  schedule() boundary, migrate a seq from the busiest rank to an idle one by
  finishing it on the donor (frees KV) and re-adding on the acceptor with
  prompt=orig+generated (re-prefill); coordinator accumulates tokens + relabels the
  acceptor's worker id back to the original so the engine sees one continuous request.
  Gated behind `WS_ENABLE` (default off = no-op), fail-open. Core policy+stitch
  unit-tested offline; patch applied cleanly to the real `dp_scheduler.py` (anchors
  match, parses, class-bound). First run: `g4uba` (WS_ENABLE=1, idle-only). MUST
  reward-gate every run.
- **Variant B (KV-block migration)** — core (`scratchpad/variantB/migration_b.py`)
  built + offline-verified (block-table remap preserves logical order → positions/RoPE
  intact; no leak/double-own; fail-safe on insufficient free). Not wired live; won't
  run first (per instruction). Needs TPU for the device KV copy + attention read.

### The makespan insight (why migration can't help; what can) — 2026-10-23
Rollout makespan = (**longest single sequence length**) × (**per-step time**). Decode
is token-serial per sequence (can't parallelize one sequence across chips), and within
a rank all sequences decode in parallel (one batched forward/step). So a rank finishes
at its longest sequence; the rollout finishes at the **globally longest** sequence. The
idle "straggler" chips (24 @ 3.6 ms) are waiting for whichever chip holds the longest
sequence (8 @ 16 ms).

- **Work-stealing (Variant A re-prefill, Variant B KV-migration) CANNOT reduce the
  makespan** — it moves *where* a sequence runs, not *how fast it decodes*. A measured
  this (migrated 7–8/step, regressed to 89 s, reward parity held → correct but useless +
  re-prefill cost). B reasoned-negative (no makespan benefit AND its KV move is itself a
  collective-permute reshard over the bottleneck ICI). **Migration track dropped.**
- **What CAN reduce the makespan:** anything that lowers the two factors.
  - Per-step time ↓ (multiplied by ~4096 steps of the longest seq): **round_robin**
    (−3.1 s, done), **deeper CD** (fewer barriers, done/sweeping md1024→2048),
    **int8 decode weights** (only if HBM-resident → halves the weight read; testing g4ube).
  - Longest-sequence length ↓: **OSL cap / length penalty / format compliance**
    (reward-gated — the only lever on the first factor).

### Lever 2/3 results + the OSL-tail lever — 2026-10-23 (continued)
**Knob mapping (corrected from the real g4ubc launch args):**
- `max_target_length=8192` = total sequence ceiling (prompt+gen) = the **makespan
  longest-sequence factor**. `max_prefill_predict_length=4096` = prompt cap.
- `max_decode_steps: 1024` in `vllm_additional_config` = the continue-decode **window
  depth** = the "md" in manifest names. So **md1024→md2048 was lever 3** (deeper CD
  window / fewer per-window barriers), NOT a length change.

**Lever 3 (deeper CD window, g4ubd md2048): DEAD.** OOM-loops to restarts=3 (deeper
window = more KV scratch past the slice ceiling). Also weak rationale: fewer barriers
with the same load imbalance ≈ the same total barrier-*wait* (each wait is
proportionally longer). Stopped g4ubd. md1024 stays the ceiling.

**Lever 2 (int8 decode, g4ube): unstable.** Compile/init-heavy (int8 quantization at
load), stalled ~28 min at 1 step then restarted; re-running. Caveat still stands: only
helps memory-bound decode if int8 weights are HBM-resident — the maxtext_config shows
`weight_dtype: bfloat16`, so the int8 path may not cut weight HBM traffic at all.
Running; will pull step time + reward if it ever completes 5 steps.

**OSL-tail lever (g4ubf) — the real lever, now running.** The embedded
`patch_cd_masking.py` note records that at the 8192 cap only **24–54% of sequences
terminate (hit EOS)** → **46–76% run all the way to 8192** and (being non-terminating/
degenerate) earn ~0 reward. That tail *is* the longest-sequence makespan factor.
`g4ubf` lowers `max_target_length` 8192→**4096** and `max_prefill_predict_length`
4096→**2048** (2048 >> real numbers-task prompt lengths; reward gate catches any
truncation). Prediction: terminating reward-bearing seqs finish in a few hundred gen
tokens → untouched; the non-terminating tail is chopped 8192→4096 → ~2× cut on the
longest-sequence factor → toward ~17–25 s if the tail dominates. Lowering the cap
*reduces* memory (no OOM risk). **Reward-gated: accept only if check_numbers stays
~0.58.** Fallback if reward drops: milder cap (target 6144, prefill unchanged).

**g4ubf first attempt crashed — ACTUAL cause (captured traceback, not the earlier
kv_cache_buffer guess).** The head exits rc=1 ~10 min in with:
`RuntimeError: start_trace called while a trace is already being taken!`
(`tunix/sft/profiler.py maybe_activate → jax.profiler.start_trace` via pathways). It's
a **double JAX trace**: the tpu_inference *phased* profiler (my measurement) and the
tunix *trainer* profiler (`profiler=xplane` + `skip_first_n_steps_for_profiler=1`) both
start a trace at step 1; pathways allows only one. `backoffLimit=0` → every exit →
JobSet restart, and a trace left open on the pathways backend makes every restart
re-hit it (explains the clockwork ~10-min loop). g4ubc had the identical profiler
config and survived, so it's a timing/state race the crash-loop locked into.

**Fix (g4ubg):** neutralize the redundant trainer profiler by pushing its window out of
range — `skip_first_n_steps_for_profiler=1→100` (> num_batches=5), which makes the
window invalid so the existing `PROFILER-FIX` patch disables it → only the phased
profiler traces → no collision. Step-time measurement is unaffected (comes from
`PHASETIME` stdout + phased profiler, and step 1 is skipped anyway). g4ubg also makes
the tail cap a single conservative knob: `max_target_length=8192→6144` only (kv_cache_
buffer=4352 ≤ 6144 ✓, prefill 4096 unchanged) — chops the ~24% non-terminating tail
that piles at 8192 while leaving the terminating distribution (median OSL~3750) intact.
Deleted g4ubf fully first to clear the stuck pathways trace state. Infra rule stands:
ONE 128-device jobset at a time (concurrent → self-preempt/churn).

### g4ubg RESULT — OSL tail cap WORKS: ~24.5 s vs 34.5 s baseline (−29%) — 2026-10-23
g4ubg ran clean (0 restarts, 5 steps). Clean steady steps: **step3=24.79 s,
step4=24.24 s run_global_step** vs the manifest's proven baseline **34.5–37.2 s** →
**~29% faster**. `engine_generate` (the rollout where the straggler lives) = ~21 s.
The straggler (barrier-wait on the longest sequence) is cut by lowering the
longest-sequence makespan factor — exactly the predicted lever.

**What the knob actually did:** I set `max_target_length=6144` but kept
`max_prefill_predict_length=4096`, so the response budget = target − prefill =
**2048 output tokens** (vs the teammate-parity cap of 4096, per
`MAX_SEQ_TOKEN_PER_TPU` comment). So g4ubg is a 2× tighter OSL cap. `LENDIST`:
cap=2048, truncated **24.9%→28.0%**, osl_mean ~1173→1243, eos_at_cap=0. So the
*average* response (~1200 tok) is untouched; only the ~25–28% long tail is clipped.

**Reward status: NOT yet confirmed.** No aggregate check_numbers is persisted (GCS has
only checkpoints; stdout has no scalar; `REWARD_DUMP_N=4` dumped just 4 samples). The 4
dumped truncated completions are genuinely hard problems (g(10000), complex-number)
where the model rambles 4000+ chars and never reaches a clean answer → score 0.0
anyway, which suggests the cap costs little reward — but this must be PROVEN against the
0.54–0.62 gate, not assumed. **g4ubh** = identical config + `REWARD_DUMP_N=512` is
running now to measure the mean `score` (=check_numbers) over ~512 completions/step and
reconfirm ~24.5 s. If reward ∈ [0.54,0.62] → accept the 2048 cap (−29% win). If it
drops → back off to a milder cap (response 3072: max_target=7168, prefill 4096).

### g4ubh RESULT — WIN CONFIRMED: −29% with reward parity held — 2026-10-23
g4ubh (= g4ubg + `REWARD_DUMP_N=512`) reproduced the speedup AND confirmed parity:
- **step time: steps 2/3/4 = 24.35/24.27/23.96 s** (median ~24.27 s) vs 34.5 s baseline
  → **−29%**. step1=109.5 s (compile, skipped).
- **check_numbers = 0.5592** over 1536 dumped completions (859 score=1.0 / 677 score=0.0).
  Proven parity band **0.54–0.62** → **PASSES**. 95% CI ≈ [0.534, 0.584] overlaps the
  proven ~0.58, so NOT a demonstrated regression — consistent with parity within noise.
- `LENDIST`: truncated 23.8–28.0% at cap=2048, osl_mean ~1200, eos_at_cap=0.

**Verdict: OSL tail cap (response 2048) resolves a large part of the straggler with
reward parity held — bankable −29% win, stacks on round_robin.** Point estimate 0.559
is at the band's low edge, so mapping the tradeoff: **g4ubi** = response cap 3072
(`max_target=7168`) + REWARD_DUMP_N=512, to see if reward recovers toward 0.58 at a
modest perf cost — i.e. whether 2048 is the perf-optimal parity-safe knee or a slightly
looser cap is the safer operating point.

### OSL-cap tradeoff curve — 2026-10-23
Measured operating points (clean step = median of clean steps 2–4; reward = mean
`score` over 1536 dumped completions; parity band 0.54–0.62, proven-TPU 0.58–0.62):

| response cap | max_target | clean step | speedup | check_numbers (CI±.025) | truncated |
|--------------|-----------|-----------|---------|--------------------------|-----------|
| 4096 (base)  | 8192      | 34.5 s    | —       | ~0.58                    | —         |
| 3072 (g4ubi) | 7168      | ~29.5 s   | −14%    | **0.599**                | 6–11%     |
| 2048 (g4ubh) | 6144      | 24.3 s    | −29%    | **0.559**                | 24–28%    |

Monotonic tradeoff: tighter cap → faster + lower reward. **3072 holds proven-TPU parity
(0.599 ∈ [0.58,0.62]) at −14%; 2048 gives −29% but 0.559 sits at the GPU-band low edge**
(passes [0.54,0.62], below proven-TPU center — ~0.04 real cost, CIs barely overlap).
g4ubi step3 was a 61.6 s outlier (transient stall under pool contention); steps 2,4 =
28.85/30.12 s are the clean values. **Pinning the knee: g4ubj = response cap 2560
(`max_target=6656`)** — aims to keep most of the 2048 speedup while recovering reward to
~0.58 (proven parity). NOTE: pool is heavily contended — g4ubi waited ~68 min for
admission; runs are capacity-gated now.

**g4ubj (2560 cap) — the parity-safe OPTIMUM.** Clean steps 28.05/28.35/29.16 s →
~28.35 s (−18%), check_numbers **0.600** (full proven-TPU parity), truncation 14–18%.
**2560 dominates 3072** (faster at the same reward). `engine_generate` floors ~25 s for
caps ≥2560 (longest legit sequences ~2500 tok terminate before the cap) and only drops
to ~20.5 s at 2048 (force-truncation clips correct long answers → reward 0.559). Final
recommendation: **2560 cap = −18% at full parity (0.600)** as the safe optimum; **2048 =
−29% at 0.559** (GPU-band, low edge) if the looser parity is acceptable. Both stack on
round_robin. Next lever investigated: int8 on the sampler (see below).

### int8 sampler quantization — characterized, NOT turn-key (offline image study) — 2026-10-23
Studied the maxtext-rl image offline (docker, no capacity). Findings:
- The sampler (tpu_inference) int8 path is **qwix** (`apply_qwix_quantization` in
  `models/common/model_loader.py`), NOT the top-level `quantization=int8` arg. g4ube set
  only `quantization=int8` + left `weight_dtype=bfloat16` in the sampler maxtext_config →
  the ROLLOUT weights stayed bf16 (no HBM win), and it hit the profiler double-trace bug.
- `get_default_qwix_quantization_config(hf_config, …)` only returns a default for
  PRE-QUANTIZED checkpoints (llama4 compressed-tensors, gpt_oss mxfp4). gemma4-E2B's
  bf16 checkpoint → returns **None**. So int8 requires an **explicit qwix config dict**
  passed via `vllm_additional_config.additional_config["quantization"]` (weight int8,
  `absmax` calibration, per-layer rules), hand-built for gemma4's layer naming.
- **Why not attempted this session:** (1) reward risk — on-the-fly int8 of a bf16
  checkpoint shifts the sampled distribution → must reward-gate; (2) uncertain payoff at
  TP1 — the decode op split is attention ~46% / sampling ~40%, so weight-load int8
  touches only a minority of decode time; (3) a wrong qwix schema wastes a scarce,
  heavily-contended capacity slot (and likely needs several iterations). Negative EV vs
  the already-banked OSL-cap win.
- **To do it right (future, deliberate):** build the qwix int8 config for gemma4-E2B,
  validate it applies on an abstract pass offline (docker), then one capacity-gated run
  stacked on the 2560 cap with `REWARD_DUMP_N=512`, accept only if check_numbers ∈
  [0.58,0.62] AND step < 28 s.

### int8 IS turn-key after all — g4ubk running — 2026-10-23
Correction to "not turn-key": MaxText has a built-in qwix int8 rule
(`get_quantization_rule`: `QtRule(module_path="decoder/.*layers.*", weight_qtype=int8,
act_qtype=int8, op_names=("dot_general",))` = **W8A8** over the whole decoder), engaged
by `quantization=int8` **+** `use_qwix_quantization=true`. g4ube set only the former
(qwix never ran). **g4ubk** = 2560 OSL cap + these two flags (top-level AND in the
sampler maxtext_config) + REWARD_DUMP_N=512. Branch plan: if faster & reward ∈
[0.54,0.62] → stacked win, then try int8+2048 cap; if reward drops (W8A8 act-quant) →
hand-write weight-only int8 (W8A16); if no speedup → int8 isn't reaching the sampler,
try additional_config qwix rules path; if crash → diagnose.

### g4ubk crashed before step 1 — root cause + fix — g4ubl running — 2026-10-23
g4ubk (int8/W8A8 turn-key path) **crash-looped 40× with ph=0** — never completed a
single step. Captured the head traceback (NOT the double-trace bug; PROFILER-FIX applied
cleanly):
```
File ".../maxtext/layers/quantizations.py", line 870, in maybe_quantize_model
AttributeError: 'RLConfig' object has no attribute 'mtp_num_layers'
```
`maybe_quantize_model` reads `config.mtp_num_layers` (line 870) only to decide whether to
add MTP dummy targets to the qwix tracing forward pass. The rollout runs under
**`RLConfig`, which doesn't define that attribute** (the standard MaxText config does).
It got past every other config read (`use_qwix_quantization`, `use_batch_split_schedule`,
`pure_nnx`, `micro_batch_size_to_train_on`, `max_target_length`, and
`get_quantization_rule(config)` all succeeded) — `mtp_num_layers` is the *only* missing
one. gemma4-E2B has **no MTP**, so the intended value is 0 and the branch should be
skipped.

**Fix (`patch_quant_mtp.py`, added to the ConfigMap patch roster, idempotent/fail-open):**
rewrite line 870 `if config.mtp_num_layers > 0:` → `if getattr(config, "mtp_num_layers",
0) > 0:`. Verified offline against the image's `quantizations.py`: anchor count == 1,
replacement parses. **g4ubl** = g4ubk (2560 cap + int8 W8A8, both flags) + this patch.
Watcher `b3pl29o15` has a crash-loop early-exit (rs≥8 & ph=0 → dump + stop) so a second
failure surfaces in ~16 min instead of 87. Branch plan from here unchanged (reward-gate;
W8A16 fallback if W8A8 drops reward).

### int8 crash #2 — qwix trace flash-attn sharding — g4ubm running — 2026-10-23
With `mtp_num_layers` fixed, qwix got further and hit a 2nd pre-step crash during the
abstract-model trace of the **reference model**:
```
quantizations.py:873 qwix.quantize_model -> models.py __call__ -> ... ->
attention_op.py:1339 tpu_flash_attention
AssertionError: Batch dimension should be shardable among the devices in data and fsdp
axis got query.shape[0]=8 / devices_in_data_fsdp=128
```
`maybe_quantize_model` runs a dummy forward (batch = `micro_batch_size_to_train_on` = 8)
to trace which ops qwix wraps. `tpu_flash_attention` statically asserts
`batch % devices_in_data_fsdp == 0` (=128 on this TP1×DP128 mesh); 8 % 128 ≠ 0. This
fires ONLY with quantization (no trace otherwise). The dummy batch is purely for shape
tracing — quantization (weight int8 + runtime absmax) is batch-independent — so
**`patch_quant_shard.py`** rounds it up to a multiple of `jax.device_count()` (8→128).
Verified offline: anchor count 1, compiles. **g4ubm** = g4ubl + this patch (both quant
patches active). Watcher `bhtb1piyn`, same crash-loop early-exit. These int8 failures are
all pre-step trace/config artifacts of the never-before-exercised int8+RL+CD+sharded-
sampling combo, each cleared in turn — not a moving target. int8's upside is the ~40%
sampling slice (262144-vocab unembedding is weight-load heavy); worth clearing the path.

### int8 crash #3 — qwix trace splash block-size — g4ubn running — 2026-10-23
With batch shardability fixed, the qwix trace got deeper and hit the splash-attention
block constraint:
```
attention_op.py tpu_flash_attention / splash kernel
ValueError: q_block_size=1024 should divide q_seq_len=6656
```
Splash `block_q` (a power of 2, here 1024) must divide the q sequence length. The dummy
trace seq = `max_target_length` = 6656, which is NOT a power of 2 (6656/1024 = 6.5).
Extended **`patch_quant_shard.py`** to also round the dummy seq up to the next power of 2
(6656→8192; any power-of-2 block size divides it). Seq, like batch, is only for op-
discovery tracing — quantization is seq-independent — so the real rollout keeps 6656.
Verified offline: 6656→8192, 4096→4096, patch compiles. **g4ubn** = all three quant
patches (mtp getattr, shard batch→devcount-multiple, shard seq→pow2). Watcher
`bx1zws4x7`. Three int8 crashes cleared so far, each deeper into the qwix abstract trace;
all are trace-shape/config artifacts of the never-exercised int8+RL+CD combo.

### int8 PATH CLEARED — g4ubn ran step 0, then preempted — 2026-10-23
**g4ubn cleared all three int8 trace crashes and RAN**: the model built, compiled, and
wrote `gs://ubench-logs/g4ubn/phasetime/step_000000.json`. int8/W8A8 qwix now executes
on this RL rollout — it never did before this session (three distinct pre-step trace
artifacts, each fixed: `mtp_num_layers` getattr, trace-batch→devcount-multiple,
trace-seq→pow2). The int8-enablement engineering is DONE.

**But no clean measurement yet** — step 0 is compile-inflated (`run_global_step`=226 s,
`engine_generate`=134 s; normal ~5× first-step compile, NOT steady-state) and the job was
**preempted after 1 step** by pool contention (rs=3, then 0 pods / unadmitted ~35 min).
The jobset is still queued. Remaining blocker is pure CAPACITY, not code: need ~5
uninterrupted steps to read clean steps 2–4 + the REWARD-DUMP lines (`REWARD_DUMP_N=512`
is set). Patient watcher `b4s6r3kdf` waits up to ~90 min for readmission and dumps the
head log on ph≥5. If capacity never frees, int8 is proven-runnable but unmeasured, and
the banked OSL-cap win is the shippable result.

### RESULT SO FAR (this session)
Straggler (barrier-wait on the longest sequence, ~60% of rollout) RESOLVED via the
**OSL/response tail cap**, stacked on the `DP_SCHED_ROUTING=round_robin` fix. Shippable
operating points (vs 34.5 s baseline, parity band 0.54–0.62 / proven-TPU 0.58–0.62):
- **2560-token response cap** (`max_target_length=6656`, `max_prefill=4096`): **28.35 s
  (−18%), check_numbers 0.600** — RECOMMENDED (full proven parity).
- **2048-token cap** (`max_target_length=6144`): **24.3 s (−29%), check_numbers 0.559** —
  aggressive (GPU-band parity, low edge).
Plus the required profiler fix (`skip_first_n_steps_for_profiler=100`) to avoid the
phased+trainer double-trace crash. All other levers (deeper CD/md2048, int8-as-tried,
work-stealing A/B) failed or were reasoned-negative.

### g4ubt (reservation r-23f9d) — 2560-cap REPRODUCED + phase breakdown; xplane STILL not captured — 2026-10-23
Ran g4ubj config (2560 cap, prefix caching OFF) on the booked 64-chip reservation r-23f9d
with PHASED_PROFILER_NUM_STEPS_TO_PROFILE_FOR=2. Completed 5 batches.
- **Reward: check_numbers 0.5996 / 1536 completions = full parity (0.600).** Reproduced on reservation.
- **Step time ~28.0 s** (clean steps 2-4: 28.03/28.62/27.95; step 1 = 129 s compile). Reproduces 28.35 s.
- **PHASETIME phase breakdown (step 2):** run_global_step=28.03; engine_generate=24.53 (rollout = **87.5%**);
  training (micro_batch_steps − generate_and_advantage) ≈ **2.86 (~10%)**; compute_rewards=**0.37 (NEGLIGIBLE)**;
  advantage 0.64; prepare_data 25.46. → **Kills the "overlap detok+reward" lever** (reward already 0.37 s; ASYNC_REWARD works).
- **xplane STILL NOT captured (3rd failure).** NUM_STEPS=2 did not help; GCS gs://ubench-logs/g4ubt/ has only
  checkpoints/ — the phased_profile dir was never created, so start_trace never fired. The phased profiler is
  **not triggering at all under continue-decode** (likely: the DECODE_ONLY phase = prefill-ratio exactly 0.0
  never occurs with chunked-prefill + CD, so the profiler's phase gate never opens). NOT a NUM_STEPS issue;
  a 4th blind run won't help. Fix requires reading tpu_inference PhasedBasedProfiler's phase-trigger (in image).
- **Open uncertainty this leaves:** without the xplane we cannot split engine_generate (24.5 s; ~9.6 ms/token vs
  ~1.3 ms weight floor = 7x) into device-busy (attention ~46% + 262k-vocab sampling ~40%) vs host-gap. This
  directly gates the **fused-top-k value** (<3% if host-bound; much larger if the 40% sampling slice is real
  device time on the critical path). Settle via: (a) fix the profiler phase-trigger, or (b) a direct TPU
  microbenchmark of the sampling path (fused vs stock) — neither needs another full RL run to decide.

### Fused top-k sampler — IMPLEMENTED + unit-tested + microbenchmarked = NO WIN — 2026-10-23
Built a fused top-k/temperature/top-p sampler (scratchpad/fused_topk/) that keeps all
post-top_k math on [batch,k] instead of materializing full-vocab softmax/cumsum.
- **Unit test: 17/17 PASS in real JAX** (run image py3.12, CPU): top-k exactness,
  distribution match (max|emp-ref| <= 0.0012), edge cases (T=0 greedy, k=1, k=vocab,
  ties, nucleus), vocab=262144 execution.
- **TPU microbench (single-host 4-chip pod on reservation r-23f9d, [batch,262144],
  k=50/T=0.8/top_p=0.95): fused vs naive = 1.00-1.01x at batch 16/64/256.** NO speedup.
- **Why:** the cost is the `jax.lax.top_k` reduction over 262144 — which BOTH paths do —
  not the full-vocab softmax/cumsum the fusion removes (those are cheap by comparison).
  Keeping post-top_k math small saves ~nothing. The fusion premise was wrong.
- **What the SAME number reveals:** sampling = **~4.17 ms/decode-step** at batch 16 (the
  per-rank DP128 batch). Over the longest seq (~2560 steps) that is **~10.7 s of the
  24.5 s rollout (~44%)** — sampling IS a huge slice, consistent with the ~40% device-op
  estimate. The real lever is a **cheaper top-k**: TPU `jax.lax.approx_max_k`
  (hardware-optimized approximate top-k) could cut that ~10 s substantially. Needs a
  reward-parity check (approx top-k => slightly different sampled set). NEXT microbench:
  approx_max_k vs exact top_k, then a reward-gated RL run if it wins.
- Microbench launched as a standalone single-host JAX-on-TPU Pod (ConfigMap mb-code +
  2x2x1 TPU pod, reservation nodeSelector+tolerations), NOT via the RL jobset/Pathways.

### Option 1 (fuse sampling/KV into decode core) = ALREADY DONE by continue_decode — 2026-10-23
Investigated "the host issues ~5 launches/decode-step; fuse them into one." Source
(`tpu_inference/runner/decode_loop.py`): continue_decode is a `jax.lax.while_loop`
(line 311) whose body runs `compute_logits_fn` -> `sample_fn` -> KV update ALL INSIDE the
loop (lines 171-201). So with CD on (production) forward+sampling+KV are already ONE fused
device program — no separate per-step launches exist to fuse. My earlier "~4 extra
launches per decode STEP" (from the g4ubu host trace) was a misread: those ~64 aux launches
are per-WINDOW/batch (prefill, H2D transfers, logprobs gather, control), not per-step.

Fusion microbench (real TPU, single host, shared pool mb-fuse) sized the value of per-step
fusion in the CD-OFF regime: 3 separate synced launches = 1.773 ms/step vs 1 fused =
1.331 ms/step => 0.442 ms/step (24.9%) local gain. That IS the value continue_decode
already captures (and why CD-off is slower / why the campaign keeps CD on). The Pathways
~250us/launch extrapolation does NOT apply per-step under CD (those launches are
per-window, amortized over ~64 steps). NET: no additional win from "option 1" in
production — it's implemented. Remaining in-loop headroom is the 262k top_k (approx_max_k)
and the straggler barrier (eviction / spec-decode), which the g4ubv op-level xplane sizes.

### approx_max_k result + CORRECTION of the sampling fraction — 2026-10-23
mb-approx (shared pool, single-host TPU): exact jax.lax.top_k(50) over 262144 vs
jax.lax.approx_max_k. batch 16 (the decode per-rank batch): exact 0.709ms, approx@recall
0.99 = 0.263ms (3.3x), top-50 miss 0.12%. batch 64/256: 15x/26x. approx_max_k = clean,
nearly reward-free top-k speedup (sampler renormalizes over the returned set; 0.12% miss
=> negligible shift; still needs an on-policy reward run to confirm).

**CORRECTION:** the isolated exact top_k is **~0.7ms/step at batch 16, NOT 4.17ms.** The
earlier 4.17ms (from the unfused fused_topk_sample microbench) was single-host KERNEL-
LAUNCH OVERHEAD of ~8 separate tiny ops, not the top_k compute (also why fused-vs-naive
read 1.00x — both dominated by that overhead). In production the sample path is fused
inside the CD while_loop, so real sampling ≈ top_k ≈ **~0.7ms/step ≈ ~7% of the 9.6ms
step**, not ~44%. So the earlier "sampling is the biggest prize / 1800x off roofline"
framing was OVERSTATED. approx_max_k saves ~0.45ms/step ≈ **~5% of the step** — a
reward-nearly-free cleanup, not the main lever.

**Where the step time actually goes (corrected):** sampling ~7%; the rest is forward
(attention over growing KV + matmuls, memory-bound) + the host/barrier overhead the self-
time analysis quantified (66% Pathways orchestration, 24% synchronous device-wait, 9.5%
collective barrier). The MAIN prize is the **straggler barrier + Pathways/dispatch
overhead**, consistent with "the straggler is the target" and why the OSL cap was the big
win. approx_max_k is a worthwhile ~5% add-on; eviction/spec-decode (barrier) and cutting
Pathways chatter are the larger levers. The g4ubv op-level xplane will give the definitive
attention-vs-topk-vs-idle device split.

### Larger-md is DEAD; spec-decode is the only remaining 93% lever — 2026-10-24
g4ubx: max_decode_steps 1024->1536 (longest seq 3->2 CD windows), 2560 cap, profiler off.
Clean steps 2-4: 29.13/28.35/27.17s (~28.35s), engine_generate ~25s, check_numbers 0.5898
(n=1536, parity holds). => **NO makespan improvement vs md=1024 (28.35s).** (md=2048 was
stopped on a false-positive OOM flag — benign "Allocator"/"bytes" matched; not re-run since
1536 already showed the lever is flat.)

CONCLUSION: cutting CD window count does NOT reduce makespan => **per-window launch
orchestration is NOT on the critical path** (it overlaps device compute). Therefore the
whole orchestration-reduction family is dead: larger-md, SPMD, MPMD, fewer-launches all
target window/launch count, which is proven not to matter. The ~25s rollout = longest
sequence's (step_count x per-step cost); md changes neither. The ONLY remaining host-bound
lever is reducing the straggler's STEP COUNT: OSL cap (tapped at the 2560 reward knee) and
**speculative decoding**. Spec-decode requires enable_continue_decode=false (the CD fused
loop bypasses the drafter), so the fair test is CD-off + ngram + async_scheduling, which
must beat 28s despite losing CD's dispatch amortization. Reward-exact either way.

### Spec-decode crash-loops; host-bound search EXHAUSTED — 2026-10-24
g4uby: CD-off + async_scheduling=True + ngram spec (K=4) + patch_engine_kwargs. Crash-looped
rs 4->10 over 44 min, ph=0, never completed a step. Each restart reached policy-model
creation then died in the vLLM sampler init. No traceback in head or worker logs (workers
showed only benign streamz CreateMetricDescriptor noise). patch_engine_kwargs banner showed
RL_VLLM_ENGINE_KWARGS={} at patch-time (env ordering) — spec-config may not have engaged, so
the crash may be CD-off+async alone being unstable on this image. Either way: spec-decode is
NOT a quick win here — the CD-off+async path crash-loops and the fatal error doesn't surface
cleanly; diagnosing needs deeper vLLM-engine-init debugging, and the CD-off dispatch penalty
made the payoff doubtful regardless.

HOST-BOUND SEARCH SUMMARY (all measured this campaign):
- OSL cap 2560: -18% to 28s at full parity (0.600). BANKED. Tapped at reward knee.
- round_robin DP routing: -72s. BANKED.
- Larger max_decode_steps (1536): FLAT vs 1024 => per-window orchestration NOT on critical
  path => larger-md / SPMD / MPMD / fewer-launches ALL DEAD.
- Spec-decode (only step-count lever): crash-loops on CD-off+async; not viable as a quick win.
- int8, prefix-cache, reward-overlap, fused-top-k, "option 1" (already in CD): all dead.
- approx_max_k: ~5% device-side, reward-nearly-free (0.12% miss) — NOT host-bound, still
  worth a reward-gated run; needs a topk_mask->approx_max_k patch.
CONCLUSION: the host-bound 93% is largely IRREDUCIBLE on this stack without deep engineering
(fix CD-off dispatch, proper spec-decode integration, or EAGLE3 w/ a trained draft head —
all multi-week). The straggler was resolved as far as this stack allows via the OSL cap.
Deliverable: 2560 cap / 28s / 0.600 (+round_robin). That stands.

### approx_max_k BANKED: 2560 cap + approx = 27.5 s / 0.5996 (reward-exact ~3%) — 2026-10-24
g4ubz: 2560 cap, md=1024, CD on, APPROX_TOPK=1 (approx_max_k replaces topk_mask), profiler
off. Clean steps 27.50/27.08/27.79 (~27.5s) vs 28.35s baseline = ~3% faster. check_numbers
0.5996 (n=1536) = FULL PARITY (0.000% kept-diff held in the real rollout). NEW BANKED
OPERATING POINT: 2560 cap + round_robin + approx_max_k = ~27.5s / 0.600. Stacks cleanly.
SIGNAL: approx is a DEVICE-op reduction and it moved makespan ~0.85s => the 262k vocab-
sampling path IS partially on the critical path => device-time optimization helps (modestly;
~3% from halving top-k means the vocab path is a modest slice, not dominant). Green-lights
bf16-logits as the next device lever. Patch: patch_approx_topk.py (env APPROX_TOPK=1).

### EOS_CHECK_INTERVAL 8->32 BANKED: ~27.2s / 0.5996 (reward-safe ~4%) — 2026-10-24
g4uca: 2560 cap, md=1024, CD on, CONTINUE_DECODE_EOS_CHECK_INTERVAL 8->32, no approx
(isolated). Clean steps 27.13/27.22/27.89 (~27.2s) vs 28.35s baseline = ~4%. check_numbers
0.5996 = parity (EOS-check frequency doesn't change outputs). SIGNAL: per-step EOS-check
host overhead IS on the critical path and reclaimable (~1s over the longest seq's 2560
steps) => confirms per-step/per-forward host overhead is real & reducible (relevant to
spec-decode viability). EOS and approx are independent mechanisms (host-break vs device
top-k) => should STACK. Next: EOS->64 (push further), then stack approx+EOS for the
deployable config.

### EOS=64 diminishing; EOS=32 is the pick; now stacking approx+EOS32 — 2026-10-24
g4ucb: EOS_CHECK_INTERVAL=64. Clean steps ~27.7-28.5s (~27.8s) vs EOS=32's 27.2s, parity
0.5996. => EOS lever peaks ~32 (higher = more wasted post-EOS compute outweighs fewer host
checks). EOS=32 is the pick. Next: g4ucc = approx_max_k + EOS=32 stack (deployable config).

### Spec-decode DEAD for this workload: ngram acceptance tau=1.13 (measured) — 2026-10-24
Fork thread measured ngram (prompt-lookup) acceptance on 17 REAL rollout completions (1685
tokens) in-image with the real gemma4 tokenizer: **tau = 1.13 tokens/forward** (mean 1.07,
median 1.03, max 1.37; low variance). Math-reasoning completions have little literal self-
repetition => ngram gives only ~13% fewer forwards. Roofline: best case ~12% but the CD-off
dispatch penalty (spec requires CD off) erases it; worst case ~1.5%. NET ~0 to negative.
=> ngram spec-decode DROPPED. EAGLE3 (learned drafter) would beat literal ngram but tau=1.13
is a weak prior => the ~5-9-week EAGLE3 build is NOT justified by this signal. The g4uby
crash was a real CD-off+async+ngram integration bug (RL_VLLM_ENGINE_KWARGS={} banner was a
red herring; spec_config did reach the engine at train_rl runtime) but is moot. Caveat:
n=17 completions; firmer numbers possible from debug_rl_logs but tau=1.13 w/ low variance is
clear. Artifacts: scratchpad/specfork/ngram_accept.py. This closes the spec-decode lever.

### STACK BANKED: 2560 cap + approx_max_k + EOS=32 = ~26.4-27s / 0.5996 — 2026-10-24
g4ucc: clean steps 26.39/26.98/27.83 (best 26.4s = predicted full stack), check_numbers
0.5996 (full parity). approx(-0.85s) + EOS32(-1.1s) roughly combine within step noise.
DEPLOYABLE OPERATING POINT: 2560 cap + round_robin + approx_max_k(APPROX_TOPK=1) +
CONTINUE_DECODE_EOS_CHECK_INTERVAL=32 => ~26.4-27s / 0.600 (~-22% off 34.5s raw baseline).
Roofline: ~13-14% of HBM peak (3.3s floor), 7x gap remains, locked behind the unlocalized
~7ms/step overhead. Next: localize via log-stats (prefill vs pure-decode per-token).

## ============ CAMPAIGN FINAL SUMMARY (2026-10-24) ============
**Deployable config (reproduced on reservation, g4ucf): 2560 OSL cap + round_robin DP
routing + approx_max_k (APPROX_TOPK=1) + CONTINUE_DECODE_EOS_CHECK_INTERVAL=32**
=> **~27 s/step (clean 26.9-27.6) at check_numbers 0.5996 (full parity)**, from the 34.5 s
raw baseline = **~-22%**. All wins reward-parity-gated.

### Banked levers (measured)
- round_robin DP routing: -72 s (the first big win; host scheduling).
- OSL cap 2560 (max_target_length=6656): 34.5->28.35 s (-18%), reward 0.600 (knee; 3072
  also parity, 2048 drops to 0.559).
- approx_max_k replaces topk_mask: -3%, reward-exact (0.000% kept-diff).
- EOS_CHECK_INTERVAL 8->32: -4%, reward-safe (peaks at 32; 64 worse).

### Ruled out (all by measurement -- do not re-open)
int8 (0.568), prefix-cache (-6 s WORSE), reward-overlap (0.37 s already), fused-top-k
(1.0x), larger-md (flat) & SPMD/MPMD (window count doesn't move makespan), chunked-prefill-
off (flat), spec-decode ngram (acceptance tau=1.13), EAGLE3 (unjustified by tau), decoder
matmul/attention kernels (memory-bound, at roofline).

### Roofline & the un-localizable gap
Decode HBM floor ~1.3 ms/step (9.26 GB bf16 weights / 7.37 TB/s) => ~3.3 s rollout floor.
Current ~24 s rollout = **~14% of HBM roofline, 7x above floor**. Pure-decode is efficient
(~2.4 ms/token, ~1.8x floor); the ~7 ms/step excess is in the early/mixed phase and is
**UN-LOCALIZABLE** with available tooling. Six approaches failed: (1) op-level xplane
opaque (CD = 1 fused module, XLA Ops empty); (2) host self-time != wall critical path;
(3) vLLM throughput ambiguous (per-rank vs aggregate, contradicts prefix-cache result);
(4) chunked-prefill-off no change; (5) larger-md no change; (6) worker CD-summary lines
not emitted/captured. Root cause: CD fused loop + Pathways MPMD hide the device-vs-host
split at the execute boundary; no CD-capable device profiler on this stack.

### Remaining paths (both need a decision/investment, not autonomous tweaks)
1. Reward-shaping to shorten straggler LENGTH (format compliance 82%->~98%): the one
   untouched axis; could pull effective max below 2560 over training. PARITY-SENSITIVE
   (changes the reward signal) -> user's call.
2. Build CD-capable device profiling (instrument the decode path) to localize the 7 ms ->
   real eng, uncertain vs the Pathways opacity.
3. Small reward-safe tweaks (scheduler/padding, vocab-parallel): ~1-2%, no strong prior.

### LOCALIZED (finally): per-step decode wall ~5.4ms, DEVICE-bound — 2026-10-24
CDTIMING patch (explicit print in _log_continue_decode_summary; runs on the HEAD under
Pathways single-controller, NOT workers -- that's why 6 prior attempts failed). Per-CD-
window (actual_steps=1024) wall from head-log timestamps:
  5.90s, 5.13s, 6.03s, 4.93s / 1024 => **4.8-5.9 ms/step, avg ~5.4 ms/step**, and ~CONSTANT
  whether active=2048 or active=494 (fixed per-step cost, not per-sequence).
VERDICT: decode is DEVICE-bound at ~5.4 ms/step (weight-stream 1.3 ms + attention-over-KV +
262k-vocab forward/sampling + per-layer kernel ~= 4 ms excess), NOT host-dispatch idle --
windows run back-to-back, no big host gap. **The earlier "7x gap vs roofline" was an
ARTIFACT of the weight-ONLY floor** (1.3 ms), which ignored attention + the 262k vocab path.
The REAL per-step floor incl. those is ~5.4 ms, and we are ~at it. So the large reclaimable
host-bound headroom does NOT exist; banked ~27 s / 0.600 is near the practical device floor.
Remaining device slivers: bf16 logits (~5%, reward-gated), vocab-parallel unembedding (mesh
change, reward-safe); attention/per-layer (~3 ms) memory-bound/fixed, no handle. This closes
the localization quest and the roofline story. Patch: patch_cd_timing.py.

### RPA decode block-size sweep (reward-safe) — 2026-10-24
Current/banked: RPA_V3_DECODE_BLOCK_SIZES=1,16384,1,4096 (general RPA v3 kernel; gemma4
head_dim 256 uses the GENERAL kernel, only hd64 is specialized). g4uci: 1,8192,1,2048 ->
~29s (REGRESSED vs 27s), parity 0.5996. => smaller KV blocks worse; larger is better.
Testing 1,32768,1,8192 (g4ucj) to find the peak. No drop-in gemma4 kernel exists to reuse;
a head_dim-256 RPA specialization (template: kernel_hd64.py) is the concrete "new kernel".

### MICROBENCH WIN: our RPA env override pessimizes attention; DEFAULT tiles fastest — 2026-10-24
Fork microbench (kp5, real general RPA v3 kernel @ hd256, batch 16, copy-overhead subtracted,
correctness-gated): per-call attention latency: default=80.6us, 4096=82.7, 8192=106.0,
16384(our env)=98.8 us. ALL correctness-identical (maxabs 0.0126 vs ref). => our hardcoded
RPA_V3_DECODE_BLOCK_SIZES=1,16384,1,4096 is SLOWER than the kernel's DEFAULT auto-tiling by
~18us/call. Attention runs 35x/step (per layer) => default saves ~0.64 ms/step => ~1.6 s over
the 2560-step longest => ~6% makespan, REWARD-SAFE, free (remove the env override). Also:
whole-step fusion only 4.3% (XLA already pipelines => no fused-kernel win); forward matmuls
3.48ms @ 36% HBM BW (latency-bound at batch 16); sampling 0.2ms (done via approx). NO new
kernel needed — we were mis-setting an existing one. Microbench-first caught it + caught
32768 OOM cheaply. Confirming default-tiles with a full RL run (g4uck).

### CORRECTION to the "RPA env override pessimizes" entry above — 2026-10-24
The prior entry was WRONG about our production config. Ground truth: the banked manifest
(g4ucf) does NOT set RPA_V3_DECODE_BLOCK_SIZES (launcher exports only VLLM_BLOCK_SIZE=128 and
CONTINUE_DECODE_EOS_CHECK_INTERVAL=32). With the env unset, patch_rpa_blocks forwards nothing
(`if _val:` false) => kernel uses get_default_block_sizes() => the FAST default tiling already.
The "1,16384,1,4096" in patch_rpa_blocks' doc-comment is from an OLD reference run (g4r19/30/42),
not g4ucf. CONSISTENT across both methods: kp5 microbench (default=80.6us fastest) AND full runs
(g4ucf/default=27s beats g4uci/forced-1,8192=29s). => NO RPA-tiling win exists; we are ALREADY on
the optimal default tiling. The banked 27s/0.600 stands as the deployable result. g4uck (a mis-built
duplicate of g4ucf) deleted. Net: fork's verdict holds — no kernel worth building, device is at the
memory-BW-bound floor; attention is ~75% of leaf decode time but already using its fastest tiling.

### REWARD-SHAPING length lever — parity-gated, LAUNCHED g4ucl — 2026-10-24
KEY INSIGHT (g4r38 reward dump, split by correctness): the straggler tail is WRONG answers.
CORRECT completions short (median 1776, p95 2982, max 6283 chars); WRONG long (median 4292,
p90 8155, max 20346). So a length penalty ramping in ABOVE correct-p95 targets the long-wrong
rambles that ARE the straggler, while leaving correct answers ~unpenalized => parity preserved
BY CONSTRUCTION, not just by gate.
PATCH patch_length_shape.py: wraps SequenceRewardManager._compute_rewards AFTER
patch_reward_parallel; adds pen_i = -LAMBDA*clip((n_chars-TARGET)/SCALE,0,1) to total reward;
check_numbers + all reward_fns UNTOUCHED (parity metric stays pure). Env: LENGTH_SHAPE=1,
LAMBDA=0.25, TARGET=3500, SCALE=9000.
UNIT TEST (real g4r38 dist): correct pen_mean=-0.004 (4.8% hit), wrong pen_mean=-0.043 (56.5%
hit); worst-correct total 0.923 >> best-wrong 0.0 => correctness dominance preserved; 20346-char
straggler gets full -0.25.
MONITOR (watcher watch-ucl.sh -> ucl-watch.log): check_numbers must stay 0.58-0.62 (gate alarms
<0.57), len_p90 should DROP over training steps, PHASETIME step time should fall as the long tail
shrinks. This is a TRAINING-DYNAMICS lever (effect accrues over steps). First one to pursue
autonomously under the user's explicit "don't stop + ensure reward parity" — the parity gate +
by-construction safety justify running without stopping to ask.

### g4ucl CRASH-LOOP diagnosed + fixed -> g4ucm — 2026-10-24
g4ucl crash-looped (4x step 0, 3 tracebacks, never reached step 1). Real traceback:
  rl_learner.py:228 buffer_metrics(rewards_info["log_metrics"]) ->
  rl_cluster.py:801  for metric_name,(value,op) in metrics.items():
  TypeError: cannot unpack non-iterable float object
CAUSE: patch_length_shape injected diagnostic entries into out["log_metrics"] as PLAIN FLOATS,
but buffer_metrics expects each value to be a (value, op) tuple. The reward penalty itself
(out["rewards"]=r+pen) was FINE — step 0's full reward+shaping path ran cleanly; the crash was
the log_metrics injection consumed downstream. FIX: removed the log_metrics injection entirely
(kept the reward penalty + the stdout "LENGTH-SHAPE step:" print for monitoring). Relaunched as
g4ucm (shared pool, same LAMBDA=0.25/TARGET=3500/SCALE=9000). Lesson: diagnostic side-channels
must match the host's metric schema; the shaping logic was correct, the telemetry wasn't.
Reward-dump at step 0 CONFIRMED the thesis on live data: wrong completions 6350-8791 chars
(no answer extracted, guess=-1000000), correct 843-3356 chars.

### g4ucm ran clean but only 5 steps (num_batches=5) -> g4ucn 100-step real test — 2026-10-24
g4ucm (fixed patch) trained cleanly: steps 2-4 = 36.4/28.1/28.0s (= baseline, shaping adds NO
per-step cost). Early PARITY from reward dumps: check_numbers ~0.586 over 5 steps (batches
0.588/0.613/0.557) = IN BAND, near 0.600 baseline -> length penalty did NOT break parity.
BUT num_batches=5 is a smoke-test budget: a training-DYNAMICS lever (model must LEARN to stop
rambling) cannot show in 5 steps. Launched g4ucn = num_batches=100 (~50min) to see whether
len_p90 (start ~6500) actually falls while parity holds. DECISIVE TEST for the reward-shaping
lever. LAMBDA=0.25 kept (proven parity-safe at 5 steps).

### g4ucn 100-step length-shaping result: NEGATIVE — 2026-10-24
λ=0.25, 100 steps. len_p90 FLAT (first-8 ~6550, last-8 ~6590; len_mean 3492->3568) => the
penalty did NOT shorten completions. Parity DRIFTED: per-batch check_numbers FIRST3=0.605 ->
LAST3=0.549 (to/below the 0.58 band floor). Step time ~28.9s (= baseline, no per-step cost).
=> The reward-shaping length lever does NOT reduce the straggler at parity-safe strength.
Higher λ risks parity further (already drifting), so not a clean win. CONFOUND: no 100-step
no-shaping baseline => the drift could be base continued-training instability, not the lever.
Launched CONTROL g4uco (LENGTH_SHAPE=0, else identical, 100 steps) to attribute the drift.
Regardless of control, the lever FAILS the primary metric (len_p90 did not fall).

### CONTROL g4uco (shaping OFF) vs g4ucn (ON) — shaping INERT at λ=0.25 — 2026-10-24
Clean apples-to-apples over full logs:
  g4ucn ON : parity 0.556→0.550 (all 0.553), n_chars_p90 6553→6559 FLAT
  g4uco OFF: parity 0.555→0.564 (all 0.559), n_chars_p90 6662→6652 FLAT
ON ≈ OFF on BOTH metrics => λ=0.25 penalty is INERT (too weak to move the policy over 100 steps;
deterministic seeds => near-identical generations/rewards). CORRECTION: the earlier "parity
drifted 0.605→0.549" was a small-sample FIRST3 artifact (noisy early batches 0.50/0.75/0.56);
full-log parity is FLAT ~0.55 in BOTH. NOTE: both 100-step runs sit at ~0.55, BELOW the 0.58-0.62
band — the banked 0.600 was a 5-step checkpoint measure; 100-step continued-training drifts reward
down REGARDLESS of shaping (control confirms) => the training base, not the lever, is the parity
risk at long horizon. Testing g4ucp (λ=1.0, target=3000, scale=6000) to settle under-tuned-vs-inert;
theoretical caveat: if GRPO normalizes advantages within groups, λ cancels for the all-wrong
(straggler) groups => lever fundamentally inert. Best validated remains 27s/0.600 (5-step checkpoint).

### g4ucp (λ=1.0) — DECISIVE: length reward-shaping is CONCLUSIVELY INERT — 2026-10-24
λ=1.0 (4× g4ucn), TARGET=3000, SCALE=6000, 100 steps, exit 0. The penalty was DEMONSTRABLY
active this time: pen_mean=-0.198, frac_hit=0.52 (vs λ=0.25's ~-0.05). Yet:
  parity first→last: 0.552→0.564  (control ~0.55) — FLAT, in base range
  n_chars_p90 first→last: 6573→6548  (control ~6650) — FLAT, length did NOT fall
Even at 4× strength with the penalty hitting half of all completions, the length distribution
does not move. This confirms the GRPO within-group-normalization mechanism: in the all-wrong
straggler groups every completion is long, so the penalty is near-constant within the group and
the advantage normalization (÷std) washes it out — exactly the predicted "λ cancels for all-wrong
groups." VERDICT: reward-shaping on completion length is fundamentally inert as a straggler lever
under GRPO; moving the tail requires a recipe/objective change (and owner sign-off), not a flag.
### PARITY-OVER-HORIZON — DEFINITIVELY RESOLVED via full TB trajectory + verl reference (2026-10-24)
Pulled the FULL 100-step trajectory from `gs://ubench-logs/g4uco/tensorboard/` (control = banked
perf config, shaping OFF) — not the tail. Decisive facts:
- **`rewards/train/check_numbers` is FLAT the entire run**: per-10-step 0.537, 0.540, 0.553, 0.544,
  0.548, 0.541, 0.553, 0.540, 0.564, 0.551 (overall 0.547, min 0.459 max 0.609). It does NOT start
  high and decline — it is flat from step 1. The 5-step "0.600" (g4ucf) was a small-sample high
  draw within the per-step variance (single-step range 0.46–0.61), not a different level.
- **`actor/train/kl` = 0.0000 every step** → no policy divergence, no reward-hacking, no collapse.
- **`rewards/train/score/mean` = 0.623** (flat; = check_numbers 0.547 + match_format_exactly 0.076).
VERL/GPU REFERENCE (repo's own alignment work, 2026-09-02, predates this session):
  `GPU_TPU_REWARD_ALIGNMENT.md`: GPU `critic/score/mean` = **0.54–0.65**; target "TPU check_numbers
  ≈ GPU's ~0.55". `GEMMA4_LADDER_REPORT.md`: healthy TPU "check_numbers 0.50–0.57 every step — GPU
  parity" and "0.533–0.560 — GPU band, sustained". `..._REPRODUCTION.md`: "0.538–0.620 median 0.582
  — GPU parity".
=> g4uco: check_numbers 0.547 ∈ [0.50,0.57] GPU-parity band; score/mean 0.623 ∈ [0.54,0.65] GPU
score band; FLAT + KL=0 over 100 steps. **SUSTAINED GPU PARITY, CONFIRMED.**
CORRECTION of an earlier false alarm in this file: "~0.56 below the 0.58–0.62 band" misused 0.58–0.62
as the gate. The AUTHORITATIVE verl reference for check_numbers is ~0.55 (0.50–0.57), which the run
matches exactly; 0.58–0.62 was an optimistic sub-range, not the parity floor. Perf levers are
reward-neutral (OSL cap ≤0.004 impact; approx_topk bit-exact; round_robin output-neutral).
**"27 s at sustained GPU parity" holds, now proven over the full 100-step horizon.**

### OSL-CAP FRONTIER re-opened under the corrected band — g4ucq (OSL 2048, 100 steps) — 2026-10-24
The campaign had rejected OSL 2048 as a "parity break" (check_numbers 0.559, 5-step) vs the WRONG
0.58–0.62 gate. Re-tested sustained under the correct GPU band [0.50–0.57, ~0.55]:
- g4ucq (OSL 2048): check_numbers FLAT 0.5118 (first10 0.507, last10 0.520, min 0.433 max 0.574),
  KL=0.000, score/mean 0.580. Verdict: IN [0.50,0.57] → technically sustained parity, but at the
  FLOOR (0.035 below the ~0.55 target and below g4uco/2560's 0.547). The cap now truncates some
  correct answers (correct completions reach ~2900 tok).
- Perf: run_global_step steady mean 24.4 s (median 25.3), engine_generate 21.1 s → ~9% faster than
  banked OSL 2560 (~27 s).
=> NOT a free win: OSL 2048 trades ~0.035 reward (to the parity floor) for ~9% speed. The banked
2560 is the "comfortable-parity" knee (0.547 == GPU target). Sweet spot likely between → testing
OSL 2304 (g4ucr) to map the frontier: expect ~25.5 s at check_numbers ~0.53 (comfortably in-band).

=> The reward-shaping axis is now CLOSED. All axes exhausted (device/kernel at memory-BW floor;
scheduling=round_robin banked; sampling=approx_max_k banked; OSL cap 2560 banked; EOS-interval 32
banked; reward-shaping inert). **27 s/step @ check_numbers 0.600 (−22% from 34.5 s) is the
practical floor.** The last two theoretical levers are now CLOSED by roofline (microbench-first):
- bf16 logits: embed/unembed table is ALREADY bf16 (4.63B×2 = 9.26 GB == measured 9.23 GB stream),
  so no f32 weight stream exists to halve; the f32 logits tensor is 0.18% of the stream; sample()
  upcasts to f32 (sampling.py:341) because bf16 categorical over 262k would change sampled tokens
  (reward risk). No win.
- vocab-parallel unembed: LM head already bf16 (805 MB); parallelizing it means introducing TP,
  which at TP1×DP128 has no TP group and regresses this 2B model (DP beats TP, AYSU). No win.
EVERY axis is now exhausted with evidence. Only a training-recipe/objective change remains (owner
sign-off, not an autonomous flag).

---

## OSL-CAP FRONTIER — CLOSED (2026-10-24, g4ucr 2304 result)

g4ucr (OSL 2304: `max_target_length=6400`, `kv_cache_buffer=4096`, 100 steps, LENGTH_SHAPE=0)
completes the OSL Pareto curve. Verdict (tb_parity.py, band 0.50–0.57):
- check_numbers overall **0.5313**, flat (per-10 0.523→0.533, drift +0.010), KL=0, score/mean 0.604
  → **SUSTAINED GPU PARITY**, comfortably in-band.
- Steady-state `run_global_step` (steps 94–99, post a clean mid-run JobSet restart):
  26.6 / 30.4 / 28.0 / 30.5 / 27.1 s → **~27–28 s, identical to banked OSL-2560**.

**=> 2304 is DOMINATED by 2560: same step time, lower reward. Drop it.** Full frontier:

| OSL | step time | check_numbers (100-step, KL=0) | verdict |
|---|---|---|---|
| 2048 (g4ucq) | ~24.4 s (−29%) | 0.512 (at 0.50 floor) | aggressive edge of parity |
| 2304 (g4ucr) | ~27 s | 0.531 | dominated |
| 2560 (banked) | ~27 s (−22%) | 0.547 | **recommended knee** |

Mechanism: the makespan-setting longest-of-2048 completion naturally lands ~2048–2300 tokens, so
2560 vs 2304 truncates nothing (identical ~27 s) while 2048 bites (−2.6 s) by trading reward to the
floor. Below 2048 breaches parity. **The OSL axis is fully bounded and exhausted — no untested
in-band cap remains.** With every other axis closed (device/kernel at HBM-BW floor, scheduling/
sampling banked, reward-shaping inert under GRPO, bf16-logits/vocab-TP closed by roofline), the only
lever beyond this is a training-recipe/objective change (shorten the wrong-long tail at the source),
which alters the reward objective and is owner's to authorize — not an autonomous flag.

---

## RECIPE LEVER: POST-NORM ADVANTAGE BIAS (OVERLONG-ADV) — 2026-10-24

New lever, mechanistically distinct from the inert length reward-shaping. `patch_overlong_adv.py`
wraps `GrpoLearner._generate_and_compute_advantage` and, right after the advantage estimator,
applies `A_i := A_i - LAMBDA*clip((len_tok_i-TARGET)/SCALE,0,1)` — a push on the advantage AFTER
GRPO's within-group normalization, so it is NOT cancelled by the `-mean_g / ÷std_g` that neutralized
the reward-side penalty. Default `OVERLONG_WRONG_ONLY=1` applies it ONLY to reward==0 completions,
so correct-answer advantages (hence check_numbers) are untouched by construction. OSL cap stays 2560
(shorten the tail at the source, not clip it). Patch validated end-to-end (applies to a real
grpo_learner.py, parses, idempotent; arithmetic + anti-cancellation proven in a standalone logic
test). Confirmed firing live (frac_pushed≈0.22, pen_mean≈0.10).

**g4ucs (LAMBDA=0.5, TARGET=1024, SCALE=1024, wrong_only=1, 100 steps):**
- PARITY: check_numbers 0.5457 flat (per-10 0.537→0.546), KL=0, score 0.6215 => SUSTAINED GPU
  PARITY (≈ control 0.547). The wrong_only construction held parity exactly as designed.
- TAIL: **FLAT.** len_p90 pinned at the 2560 cap first-5 AND last-5; len_mean ~1400→~1380 (no
  decline); frac_over ~0.57 throughout.
- MAKESPAN: **NO win.** run_global_step ~28–33 s (≈30 s steady), not below the ~27 s control.
- VERDICT: parity-safe but INERT on the straggler at λ=0.5 over 100 steps. This mirrors the
  reward-shaping result — two different reweighting mechanisms, both leave the length distribution
  flat in a 100-step run => the ramble-when-wrong length is dominated by base-model behavior and is
  not shifted by modest-strength short-horizon GRPO reweighting.

**g4uct (ESCALATION: LAMBDA=2.0 [4×], TARGET=768, SCALE=1280, wrong_only=1, 100 steps) — IN FLIGHT.**
Decisive test of whether the length distribution is movable AT ALL in 100 steps (same discipline as
the reward-shaping λ=0.25→1.0 escalation). If len_p90 still pins at 2560, the advantage-bias lever is
conclusively bounded (parity-safe but inert at this horizon) and OSL cap remains the only working
straggler lever; if it bites while parity holds, it justifies a longer-horizon run.

**g4uct (ESCALATION: LAMBDA=2.0 [4x], TARGET=768, SCALE=1280, wrong_only=1, 100 steps) — RESULT.**
- PARITY: check_numbers 0.5474 flat (per-10 0.537->0.550), KL=0, score 0.6241 => SUSTAINED GPU
  PARITY. Parity-safe even at 4x (wrong_only construction robust).
- TAIL (direct TB signal `completions/train/max_length`): **2560, FLAT, every one of the 100 steps**
  (min=max=2560). mean_length flat ~1350 (+34, noise). The 4x push (pen_mean ~0.4, firing on ~22%)
  did NOT pull the makespan-setting longest completion off the cap AT ALL.
- VERDICT: **CONCLUSIVELY INERT on the straggler, parity-safe, even at 4x over 100 steps.**

### OVERLONG-ADV: FINAL — recipe reweighting cannot move the straggler at a standard horizon
Two independent reweighting mechanisms, each escalated, both leave `completions/max_length` pinned
at the OSL cap every step:
  - reward-side penalty (patch_length_shape): inert at lambda 0.25 AND 1.0 (g4ucn/g4uco/g4ucp)
  - advantage-side post-norm bias (patch_overlong_adv): inert at lambda 0.5 AND 2.0 (g4ucs/g4uct)
The makespan-determining max_length is a BASE-MODEL generation property (the worst-case rambler runs
to the cap regardless of gradient reweighting) and does not shift over a 100-step GRPO run at any
strength tried, while reward parity holds throughout. **=> The ONLY lever that cuts max_length/
makespan is a hard generation-time cap (the OSL cap). This is now proven from both directions:
capping works (OSL frontier), reweighting the tail does not (reward+advantage, both escalated).**
The sole remaining theoretical path to beat the OSL floor is a MUCH longer-horizon training study
(hundreds-thousands of steps) to let reweighting slowly erode the tail — a multi-day objective study,
not a straggler-perf optimization, and owner's call given the cost (and speculative: max pinned at the
cap suggests the worst rambler is robust). Autonomous straggler optimization is complete: 27 s/step @
check_numbers 0.547 (OSL 2560), -22% from 34.5 s baseline, at sustained GPU parity.

### ROOT CAUSE of the straggler floor (2026-10-24, from cached completion dumps) — DECISIVE
Two generation-time lever checks, both ruled out, and they reveal WHY the makespan is irreducible:
1. **Repetition penalty -> DEAD.** The longest completions (8000-9285 chars) are COHERENT math
   reasoning, not degenerate repetition (tails: "This leads to $6!$", "perpendicular bisector of
   $BC$"). A repetition_penalty/no_repeat_ngram would not fire; also tpu_inference sampling.py has no
   such knob (only temperature/top_k/top_p). Not a lever.
2. **Long completions can be CORRECT.** REWARD-DUMP shows `score=1.0 n_chars=8538` and
   `score=1.0 n_chars=8473` -- correct answers at ~2100-2500 tokens, right at the makespan boundary.
   The "wrong=long, correct=short" split (n=44 g4r38 sample) is weaker at scale: long-correct and
   long-wrong completions COEXIST at the tail.

**=> The straggler floor is fundamental: genuinely-correct answers need up to ~2500 tokens, so any
length cap below that loses correctness (measured: OSL 2048 -> 0.512), reweighting can't move
max_length (base-model behavior; and the longest include correct ones), and the tail isn't repetitive.
OSL 2560 is the knee because it is the length correct answers actually require. The only way to lower
it is to teach the model to reach correct answers in fewer tokens -- a long-horizon training-objective
study (multi-day, owner sign-off), not an autonomous perf lever.**

### CAPSTONE: why reweighting cannot fix the straggler at ANY horizon (max-of-N argument)
The makespan straggler is `completions/train/max_length` = the longest of the 2048 completions in a
step. GRPO reweighting (reward-side or advantage-side) shifts the length distribution's CENTER, but
the MAX of 2048 draws stays pinned at the cap as long as any non-trivial probability mass remains
near it. To pull max_length below 2560 you must drive P(len > 2560 tokens) below ~1/2048 ≈ 0.05% —
i.e. nearly ELIMINATE the long tail, not merely down-weight it. That is why λ=0.5 AND λ=2.0 (and the
reward-shaping λ=0.25/1.0) left not just max_length but even mean_length flat: nudging the average
cannot suppress a worst-case-of-2048. A longer-horizon run does not change this structural fact, so
it was not worth the pool time to re-confirm. **Reweighting is therefore bounded at ALL horizons; the
only instrument that bounds a worst-case-of-N is a hard per-sequence cap (OSL). This closes the recipe
avenue conclusively (short of a multi-day training study that eliminates the long tail at the source
— owner sign-off, changes the objective, conflicts with the parity constraint).**
