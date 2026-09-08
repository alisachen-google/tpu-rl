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

---

## Window double-buffering realized via `async_scheduling` (2026-09-04)

**Discovery:** the window double-buffer I proposed to hand-write is already vLLM's
`async_scheduling=True` (env `DL_ASYNC`) — it dispatches window N+1's device work while
window N's outputs are still processed on the host. The code carried a warning that
async-scheduling + continue-decode "degrades generation quality: sequences fail to
terminate and run to their token limit" (`tpu_runner.py:864`).

**Key insight:** that failure *is* the corrupted-embedding symptom we fixed — a model
that can't emit its stop token runs to the cap under any scheduler. The warning predates
the checkpoint fix.

**Unit test (local):** on-device EOS retirement (interval=8>0) makes generated length
depend only on the true EOS position, not on host-processing lag — so async-scheduling
cannot cause runaway. PASS.

**Termination verify (g4r61v, 2×2×1, async on, fixed ckpt):** step-0 LENDIST
`eos_at_cap=0, osl_mean=1372, with_eot=0.6%, truncation 11%` — **indistinguishable from
the sync baseline; no runaway.** The warning does not survive the checkpoint fix.

**Priority full run (g4r61):** champion win512 stack + `async_scheduling=True`, big slice —
measures the perf win (rollout 31.5 → target ~8.5 s ⇒ global ~36 → ~13 s), rewards
expected unchanged. Launched; results pending.

Also fixed a harness bug found here: the launcher's `DEBUG-HOLD` slept 1200 s after every
failure, turning transient infra crashes into 20-min stalls; now env-gated
(`DL_DBGHOLD`, default 0 = fast restart). Added a `PHASE-LOG` banner before train_rl to
localize checkpoint-load vs decode failures.

---

## async_scheduling perf RESULT (g4r61, win512, 2026-09-05)

Measured, clean steps 3/5: global **34.5 / 36.0 s**, rollout 30.3–31.8 s — vs g4r55
(async OFF) 34.5–37.2 s / rollout 30.8–33.0. **async_scheduling is perf-neutral-to-~1 s
at win512, NOT the ~20 s double-buffer win.** Termination stayed healthy (the checkpoint
fix mooted the degradation warning).

**Why it's small — as predicted:** async_scheduling overlaps window N+1's device dispatch
(~0.65 s/window) with window N's host processing, so its ceiling is the total device time
(~5 s). But host processing (~3 s/window) dominates and remains on the dispatch critical
path. Hiding device-under-host saves only the device slice, which at win512's 8 windows is
within measurement noise.

**The real double-buffer win still requires the deferred-host-processing reorder** (make
window N's detok/extract/marshal a trailing pipeline so it never gates dispatch of N+1),
targeting rollout ~8.5 s. That is an engine-core change, not a config flag — the flag was
the cheap thing to rule in/out first, and it's now ruled: safe, but not sufficient.

**Net recommendation:** keep `async_scheduling=True` (free ~1 s, accuracy-safe) in the
production config, but the headroom to ~13 s/step lives in the host-pipeline engine change.

---

## Roofline gap on the current best (g4r68 20-step, 2026-09-08)

Fresh component medians (clean steps): rollout 31.4 · trainer 3.8 · sync 0.45 · rewards 0.2
· **global 35.7 s** (best 35.2). Roofline ≈ 5.7 s → **6.3× gap, ~27 of ~30 s is rollout
host overhead.**

| Phase | measured | roofline | gap | status |
|---|---:|---:|---:|---|
| rollout | 31.4 | 4.4 | 7.1× | serial-head per-window IPC — THE opportunity |
| trainer | 3.8 | 0.8 | 4.7× | small absolute |
| weight-sync | 0.45 | 0.5 | 1× | at floor |
| rewards | 0.2 | 0.2 | 1× | at floor |

Rollout = 5.2 s device (8 windows × 646 ms, HBM-bound) + **~26 s serial host** (~3.3 s/window:
gather → stop-scan → re-pack → 2048 per-request dispatch IPCs).

**Prioritized (only rollout matters — trainer/sync/rewards collectively <4 s):**
1. **Window double-buffer w/ deferred host-processing** (engine reorder, NOT the
   async_scheduling flag): rollout 31→~9 s ⇒ **global ~13 s**. Biggest single lever.
2. Batched request submission (1 bulk add_request vs 2048 IPCs/window).
3. Parallel per-window output processing (fan the 2048-row scan across 44 cores).
4. Fewer windows via shorter OSL (free tailwind — RL training shortens completions).
5. Multiprocess-DP (removes the head; Pathways-gated; touches the roofline itself).

Do NOT spend on trainer/sync/reward levers or more env knobs (TMBS/HBM/mnbt) — all null at
this window. The entire remaining headroom is host-parallelism in the rollout window driver.