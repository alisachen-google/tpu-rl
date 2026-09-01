# Gemma4-E2B RL — Complete Optimization Catalog

**Updated:** 2026-09-01 · Companion to `GEMMA4_HANDOFF.md` (ops) and `GEMMA4_PERF_REPORT.md` (history).
Reference points: our champion base (tp2×dp32 half-slice) global median ~125 s;
**tianyu-rl-g4-003 (tp1×dp128 full-slice) global ~51 s warm** — the target stack.

Control-plane companion: [`PATHWAYS_DP_BOTTLENECK_SUMMARY.md`](PATHWAYS_DP_BOTTLENECK_SUMMARY.md)
summarizes the DP scheduling and Pathways parent-process bottleneck with diagrams.

Legend: ✅ adopted (in our base) · 🏆 win (adopt) · ⚪ neutral (dropped) · ❌ rejected/broken · 🔜 to port from g4-003 · 🧪 in flight

## 1. Generation correctness (gates all perf numbers)

| Opt | Status | Evidence |
|---|---|---|
| Prefill padding-bucket fix (`VLLM_TPU_BUCKET_PADDING_GAP=32768`) | ✅ | echo 72%→39%, perf-neutral; g4r9r |
| Proper padding fix (`patch_padding_bucket`, include_max_token_size, PR #3466-style) | 🔜 | g4-003 carries it; cleaner than the env |
| **Gemma-3-STYLE textual prompt** (`gemma_data.py` custom process_data) | 🔜 | ckpt's canonical `<|turn>` ids are UNTRAINED → rambling; teammate smoke 39/39 with textual markers |
| **Stop-token union [1,106,50]** (`patch_stop_tokens` + `patch_runner_eos`) | 🔜 | without: 99.4% of completions ran to the 8192 cap |
| CD deep-window correctness: `patch_cd_masking` v2 (idempotent masked slots) + `patch_cd_report` (last non-pad row) | 🔜 | prerequisite for windows >64; tianyu run-10 was VOID without |

## 2. Rollout / decode

| Opt | Status | Evidence |
|---|---|---|
| Mesh dp=32 via tp2 (half slice) | ✅ | rollout 97→93-95; dp is the real knob (host-bound) |
| **CD window 512** (+`patch_cd_proper` PR #3374) | 🔜 | g4-003 rollout 40 s; window 64→512 = 8× fewer head round-trips. Our window-128 "neutral" verdict predated the correctness patches |
| `patch_cd_noglobal` (drop per-iter cross-DP EOS reduce) | 🔜 | 216 µs/iter × window depth |
| CD_NO_EOS_REDUCE (our variant) | ✅⚪ | neutral alone at window 64 |
| Vocab-sharded sampling + padfix (`patch_sharded_sampling`, SS_PADFIX) | ✅ | device win; wall-neutral (collectives 1.4%) |
| RPA v3 block-size forwarding (`patch_rpa_blocks`) | 🔜 | env was silently inert on MaxText path |
| `VLLM_BLOCK_SIZE=128` pin (`patch_block_size`) | 🔜 | removes engine-shape confound |
| Prefix caching OFF | ❌→🔜 re-test | ours: median win, mean regression (KV spikes) at window 64; g4-003: −6 s/step byte-identical at window 512 |
| DP round-robin routing (`DP_ROUTING=rr`) | ❌→🔜 re-test | ours: 2× regression at dp64; g4-003 uses it at dp128 (composed stack) |
| `CONTINUE_DECODE_EOS_CHECK_INTERVAL=8` | 🔜 | valid only with cd_report fix |
| Keep KV across weight sync (`patch_sync_keepkv`) | ✅⚪ | −2 s sync, stable |
| fp8 KV cache | ❌ | flax_nnx/RPA path lacks fp8 → silent hollow steps |
| TPU_MULTIPROCESS_DP=1 | ❌ | hard-gated: "not supported on Pathways" |
| tp4×dp16 (half slice) | ❌ | 128 seqs/engine collapse (corrected gen) |

## 3. Host-work overlap (reward/detok/logprobs)

| Opt | Status | Evidence |
|---|---|---|
| Parallel reward pool (`patch_reward_parallel`, 64 procs) | ✅ | 30 s→~8-10 s median |
| Per-item verify timeout (`patch_verify_timeout`, VTMO=5) | ✅ partial | per-item works; batch aggregate still spikes (g4r14b step3 128 s) |
| **ASYNC_REWARD — score during rollout** (`patch_async_reward`) | 🔜 | g4-003 rewards **0.3 s** (vs our 8-10 + spikes) |
| Fast `\boxed{}` extraction (`patch_fast_extract`) | 🔜 | kills O(len×braces) scan (~3 s) |
| **Format-guard** (`patch_g4_format_guard`, no-regex find-chain) | 🔜 | catastrophic backtracking on gemma echo: 793 s/32 items |
| Async detok + logprobs (`patch_async_detok`, `patch_async_logprobs`) | 🔜 | overlapped with decode tail |
| Vectorized EOS trim (`patch_host_extract`), batch detok (`patch_fast_detok`) | ✅⚪ | small |

## 4. Trainer / weight sync

| Opt | Status | Evidence |
|---|---|---|
| **jit_unroll** (`patch_jit_unroll`, 1 dispatch vs ~290) | 🏆 | g4r16a clean steps 113.5-123.8 vs 117-128.6 (~−4 s) |
| **Flash/splash trainer attention** (`DL_ATTN=flash`) | 🏆 | g4r16b clean 111.0-119.7 (~−3-6 s); also the OOM-avoidance keystone |
| remat_policy=minimal | ⚪ | g4r17a within noise; backward not recompute-bound |
| GC gating (`patch_gc_sync`) | ✅ | +1.2 s/step removed |
| numpy-first packer (`patch_packer_fast`) | ✅ | −4.7 s dequeue |
| Packing budget 8192 (`patch_packing_knob`) | ✅ | the "-packing" image win |
| int8 quantization | not attempted | RL-quality risk |

## 5. Full-slice serial mode (128-device trainer)

| Opt | Status | Evidence |
|---|---|---|
| TFRAC=SFRAC=1.0 at default HBM 0.6 | ❌ | train step 80.28G ask vs 36.7G free (g4r12r1) |
| + HBM 0.25 + TMBS 4 | ❌ | step 0 fits; **step-1 sync-boundary OOM** (g4r12r2, g4r18 — tp-independent) |
| **g4-003 recipe: HBM 0.3 + flash + TMBS 8 + chips_per_vm=8 + kv_cache_buffer=4352** | 🧪 **g4r19** | g4-003: 10/10 steps, no OOM, global ~51 s |

## 6. Platform / infra

| Opt | Status | Evidence |
|---|---|---|
| XLA proxy-flag ladder (dvfs_p_state=7, SparseCore collective offloads, latency-hiding scheduler, ~30 flags) | 🔜 | g4-003 passes them as pathways-proxy args (client-side LIBTPU is inert under proxy) |
| Owned quota (ubench-regression-tests, cmcs-ai-ninja-team) | ✅ | no preemption |
| maxRestarts>0, head anti-affinity | ✅ / 🔜 | ride out capacity gaps; measurement noise |
| async_scheduling=True | ❌ | changes sampling stream (VOID) |

## 7. Strategic (not yet engineering-ready)

- **Head parallelization** — multiproc-DP exists but Pathways-gated; g4-003 sidesteps via deep CD windows (fewer round-trips) rather than parallel head.
- **Vocab-chunked cross-entropy** — would cut the ~69G logits term; unlocks TMBS≥8 without flash dependency.
- gemma4_prs.diff (multimodal parity stack) — mounted in g4-003 but skipped; not perf.

## Key composition lesson

Isolated tests undersell composed stacks: window-128 was "neutral" and rr-routing "2× worse" **before** the CD correctness patches + async host work existed. g4-003's 51 s is the composition. Attribution order for porting: (1) stop-tokens+prompt (correctness), (2) CD window 512 + correctness patches, (3) async reward/detok, (4) full-slice flash trainer, (5) XLA ladder.
