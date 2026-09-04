# Gemma4-E2B RL — Headroom Analysis: Roofline vs. Measured, and the Experiment Map

**Date:** 2026-09-01 · Companion to `GEMMA4_PERF_REPORT.md` (history), `GEMMA4_OPTIMIZATIONS.md` (catalog).
Hardware: TPU v7x 4×4×4 (64 chips = **128 devices**; per device ≈ 2.3 PFLOP/s bf16, ≈ 7.37 TB/s HBM,
ridge ≈ 313 FLOP/byte). Workload: GRPO, 2048 completions/step (256 prompts × 8), packed 8192-token rows.

---

## 1. The roofline (what the hardware would allow)

Per training step at the measured generation contract (osl_mean ≈ 3,200; ≈ 6.55 M generated tokens,
≈ 6.8 M trained tokens):

| Phase | Physics | Ideal time |
|---|---|---|
| Rollout decode | ~3,200 iterations/engine; per iteration each engine reads active weights (~5.2 GB bf16) + KV (~0.7 GB) ≈ 6 GB → 0.81 ms at the memory roof | **≈ 2.6 s** (+ ~1 s prefill) |
| Rewards | pure host work, fully overlappable with decode | **≈ 0 s** (hidden) |
| Trainer | ≈ 1.06e17 FLOPs over 128 devices (≈ 40 % MFU achievable at this size) | **≈ 0.4–1 s** |
| Weight sync | one fsdp→engines reshard, ~5.4 GB | **≪ 1 s** |
| **Ideal step** | | **≈ 4–6 s** |

The decode *kernel* already sits on the memory roof (AI ≈ 15, ~100 % of its ceiling) — per-device compute
is not the limiter anywhere.

## 2. Measured (where we actually are)

| Config | Clean step | Median | Phase split (clean) |
|---|---|---|---|
| tp2×dp32 half-slice + flash/jit_unroll (our best) | **111–119 s** | ~119–125 | ro ~89–93 · rw 5–10 · tr ~15–18 |
| tp1×dp128 full-slice (g4r19) | 148–155 s | 175.6 | ro 125–130 · rw 8.5–17.5 · tr ~15 |
| tianyu g4-003, same mesh as g4r19 (ported target) | ~50 s | ~51 | ro ~40 · rw 0.3 · tr ~11 |
| **Roofline** | **4–6 s** | | |

**Total headroom: ≈ 30–50×.** Devices are ~2–3 % busy during rollout (measured: ~52 ms device work in a
~2.5 s wall window at dp128; xplane/Perfetto show `device_get` ≈ 546 ms vs 15 ms device per window at tp1,
host 97–98 % of wall on tp2/tp4).

## 3. Headroom decomposition (who owns the gap)

| # | Headroom | Size (tp1×dp128 basis) | Mechanism |
|---|---|---|---|
| H1 | **Head-loop round-trips** | ~85–105 s of the 125–130 s rollout | window-64 CD → ~50 windows/rollout × ~2.4 s host cost (single-core Python head serially servicing 128 engines through the Pathways proxy: `device_get`, scheduling, collect). Host window cost scales with dp: 0.78 s@64 → 1.0 s@32 → 2.4 s@128 |
| H2 | **Wasted decode from broken stop contract** | ~25–35 s of rollout + corrupted training signal | truncation 64→78 % rising, osl 2745→3304: canonical `<|turn>` markers are untrained for this ckpt — model neither understands the prompt nor emits turn-end ids (E1 proved stop-ids alone are void) |
| H3 | **Sequential host post-processing** | rw 8.5–17.5 s median | detok + logprob extraction + reward scoring run after decode, single-threaded head |
| H4 | **Reward tail spikes** | +80–120 s on ~1 of 7 steps (mean ≫ median) | hard sympy batches serialize 32 items/proc even with per-item timeout |
| H5 | **KV-wave rollout spikes** | +100–170 s on ~2 of 7 steps | KV pool pressure waves; worsened by squeezed `hbm_utilization=0.25` |
| H6 | **Trainer inflation** | ~14 s (15 vs ~1 ideal) | TMBS=4 doubles micro-passes (memory-forced); dispatch overhead; small-model MFU |
| H7 | **Device/collective inefficiency** | second-order today | default XLA flags; no SparseCore offloads, no dvfs boost |
| H8 | **Architecture ceiling** | the residual ~10× below ~50 s | single head process is still serial even at window 512; multiproc-DP is Pathways-gated |

## 4. Experiment → headroom map (the ladder, with expected trajectory)

Base: tp1×dp128 full-slice, OOM fix validated (flash + chips_per_vm=8 + kv_cache_buffer=4352), median 175.6 s.

| Rung | Run | Attacks | Expected effect | Projected clean step |
|---|---|---|---|---|
| E2 (running) | g4r21: CD window 512 | **H1** | windows/rollout ÷8 → host round-trip cost ÷~8; early signal: cold rollout 660→256–320 s; no OOM at depth 512 | 175 → **~95–110** |
| E7+E1 | g4r22: gemma-3-style prompt + stop-ids [1,106,50] | **H2** | truncation → low, osl ↓ ~20–30 % → proportional rollout cut + honest RL signal (tianyu: 99.4 % capped without it) | → **~75–90** |
| E4 | g4r23: async detok + logprobs (`DL_ADETOK/DL_ALOGP`) + fast extract | **H3** | post-processing overlapped with decode tail | → **~70–82** |
| E4b | g4r24: async reward (score during rollout) | **H3+H4** | rewards 8.5–17.5+spikes → **~0.3–1 s** (tianyu-proven); kills the reward tail at the source | → **~62–72** |
| E8 | g4r28: chunked per-token logps (`DL_LOGCH=1024`) | **H6, enables H5 relief** | logits 34.4→4.3 GB at TMBS=8 → trainer memory freed → can raise `hbm_utilization` back up (KV headroom vs waves) | enabler |
| E5 | g4r25: TMBS=8 | **H6** | micro-passes ÷2 → trainer ~15→~9–11 s | → **~56–66** |
| E6 | g4r26: XLA proxy-flag ladder (dvfs_p_state=7, SparseCore collective offloads, latency-hiding scheduler) | **H7** | device+collective efficiency; part of tianyu's 51 s | → **~52–60** |
| E27 | g4r27: composed toggles (prefix-off −6 s, rr routing, eos-interval-8) | **H1 residual** | only valid on the deep-window stack (isolated tests mislead) | → **~48–55 ≈ tianyu parity** |

Gate at every rung: keep iff clean-step median improves; best-so-far carried forward. Runs are 8-step
probes; a 2× confirm precedes any champion declaration.

## 5. After parity: the residual ~10× (H8)

Even tianyu's 51 s is ~10× off the 4–6 s roofline — all remaining gap is the serial head architecture:

1. **Head parallelization** — `TPU_MULTIPROCESS_DP` exists in tpu-inference but is hard-gated
   ("not supported on Pathways"). The precisely-scoped upstream ask: Pathways-compatible per-rank
   engine processes; alternatives: process-pool `_process_continue_decode_outputs`, or rollout
   off-Pathways.
2. **Deeper windows still** (1024+) — diminishing returns as per-window host cost begins to dominate
   over count; needs measurement at 512 first.
3. **Mesh revisit under the composed stack** — dp32's head fan-out is ⅓ of dp128's; back-porting the
   winning host-side stack to tp2×dp32 half-slice may beat the full-slice line (planned as the
   closing comparison).

## 6. Risk register

- **Cluster instability** (today's tax): borrowed-quota reclaim + node churn evicted E2 13× before a
  clean window; mitigations in place: capacity-gated relaunch, maxRestarts=50, off-peak retries.
- **Prompt change (E7) alters the RL problem framing** — semantically flagged; tianyu's teammate
  smoke (39/39) and our LENDIST/reward bands are the acceptance gates.
- **Deep windows + correctness**: our roster already carries cd_proper/cd_masking/cd_report/cd_noglobal;
  zombie/truncation censuses in LENDIST guard every rung.


---

## Roofline v2 (2026-09-04): healthy-model numbers

Recomputed with the fixed checkpoint's measured distributions (OSL mean ~1450 and
falling, truncation 3–10%, natural stops):

| Component | Roofline | Basis |
|---|---|---|
| Decode (straggler-bound) | ~4.4 s | 1.26 ms/iter (9.3 GB weights / 7.37 TB/s per device, TP1 replica) × ~3500 straggler iters |
| Trainer | ~0.8 s | 6 FLOPs/param/token over 2048×~1600 tokens @128 dev, 40% MFU |
| Prefill | ~0.02 s | 155-token prompts, compute-bound |
| Sync + misc | ~0.5 s | measured |
| **Total** | **≈ 5.7 s/step** | |

Gap: best measured 56.2 s (win64) = **9.9×**; expected ~40 s (win512, g4r55) = **7.0×**.
The entire gap is serial-head window overhead (~0.75–1 s/window × 23–64 windows @win64).

### Levers ranked (healthy-model era)

| # | Lever | Type | Expected | Status |
|---|---|---|---|---|
| 1 | Window 512 on our launcher | code (root-cause the cd512 crash) | −11 s+ and compounds with L2 | standing defect; g4r55 gets it via reference stack |
| 2 | **Prefill rebalance** MPPL 4096→1024 | env | smaller prefill graph + **response cap 4096→7168 free** (truncation→~0) | **g4r59 staged** |
| 3 | **Engine batching** mnbt 8448→33280 | env | fewer scheduler iterations per window (champion's value) | **g4r60 staged** |
| 4 | Window double-buffering (dispatch N+1 during host-process N) | code (tpu_runner) | hides most per-window host cost | next engineering item |
| 5 | Length-aware prompt bucketing | code | shrinks straggler window count (OSL variance) | backlog |
| 6 | Natural OSL decay | free | RL training itself shortens completions (1750→1350 in 8 steps) — steps speed up over training | observed |

Experiments g4r59/g4r60 auto-run after the g4r56–58 micro-ladder (roofline-chain.sh).