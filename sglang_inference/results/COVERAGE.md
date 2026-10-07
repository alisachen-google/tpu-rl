# Coverage review — edge cases and correctness (plan v2 vs. this build)

Reviewed 2026-10-07 against sglang-torchtpu main `09d1244c26`. "Plan" = the v2 test plan; "Tool" = what we have in `~/pd_disagg/scripts`.
Verdict up front: the plan's three correctness gates (differential, structural, semantic) are the right skeleton, but as written they
miss (a) a noise floor for the differential gate, (b) an absolute accuracy anchor, (c) PD-specific state machines that only break under
memory pressure (decode retraction, decode pre-allocation vs. prefill admission), (d) sampling-parameter plumbing across the P→D boundary,
(e) the very-short-prompt regime for the sliding-window pool, and (f) the built-in KV integrity checks this build already ships.
Items marked **ADD** are new; **HAVE** are covered by plan + tools; **PART** are partially covered.

## 1. Correctness oracles

| # | Oracle | Status | Notes / what to run |
|---|---|---|---|
| 1.1 | Monolithic golden on the same hardware, same resolved flags | HAVE | `G` recipes (fp8 and bf16 KV). Goldens for 1k–128k done (gpt-oss, Qwen). 256k–1M need the `1m`-shape golden. |
| 1.2 | **Determinism noise floor** (golden vs golden, c=1 vs c=2) | **ADD** (running) | Without it a differential mismatch cannot be attributed. Running now on gpt-oss (`golden/gpt-oss_fp8_rep2_c1/determinism.txt`). If greedy outputs differ run-to-run, strict gates must be applied at concurrency 1 only. |
| 1.3 | **Absolute accuracy anchor**: GSM8K-100 (the repo's own CI gate for gpt-oss PD is 0.80 ± 0.03) on golden and every PD config; MMMU-Pro subset for Qwen3-VL | **ADD** (running for golden gpt-oss) | Catches "golden and PD agree but both are wrong" (kernel bugs hit both). `python -m sglang.test.run_eval --eval-name gsm8k --num-examples 100 --api chat`. |
| 1.4 | Independent CPU/HF reference for short prompts (1k, 8k) | **ADD (optional, Day 3+)** | Both hosts have 944 GiB RAM; HF bf16 CPU for 32 output tokens is feasible for gpt-oss (dequantised MXFP4) and Qwen. Only worth it if 1.3 shows an accuracy gap. |
| 1.5 | Built-in **KV checksum across the transfer** (`--disaggregation-enable-kv-checksum`, Adler-32 on prefill, verified on decode; mismatch aborts the request) | **ADD** | Run TC-1 and TC-6 once with it ON (both engines) as a diagnostic variant: silent KV corruption becomes an explicit abort. |
| 1.6 | Built-in **KV canary** (`--kv-canary log|raise`, `--kv-canary-real-data partial`, `--kv-canary-sweep-interval N`) | **ADD** | Detects use-after-free / overlapping page writes in the pool. Enable (`log`) during TC-2 eviction-under-transfer, TC-3 overflow, and the retraction test. |
| 1.7 | Token accounting cross-check (`/metrics` prompt_tokens_total on prefill == Σ prompt_tokens; generation_tokens_total on decode == Σ completion_tokens; `prompt_tokens` identical between golden and PD) | **ADD** | Cheap; a mismatch means a request was double-counted, dropped, or re-tokenised differently on decode. |
| 1.8 | Structural: finite logprobs (all ≤ 0, no NaN/Inf), exact `completion_tokens`, no empty output, 4-gram loop | HAVE (PART on logprobs) | `golden_diff.py diff` checks text/NaN markers; extend to assert every output logprob is finite and ≤ 0 (small follow-up). |
| 1.9 | Semantic needle 10/50/95 % within native context | HAVE | `verify_pd_disagg.py`, `build_prompts.py` embeds 3 needles per prompt. |

## 2. PD state machine / memory-pressure cases (highest bug yield, under-covered in the plan)

| # | Case | Status | How |
|---|---|---|---|
| 2.1 | **Decode retraction** (`--disaggregation-decode-retraction-backup cpu_tensor` is the TPU default): decode pool full → requests retracted to CPU tensors and restored | **ADD** | Config A with `max_running_requests 8` at 128k: 8 × 132k > 786k pool guarantees retraction. Gate: every output == golden (c=1 noise floor), no leaked pool after drain. Also with EAGLE3 (Config D) and with decode radix (E). |
| 2.2 | Decode pre-allocation vs. prefill admission: prefill accepts N requests whose ISL+OSL exceed decode free pool | PART | Plan has "slow decode / backlog". Add explicit check that `num_queue_reqs` on decode drains in order and prefill never hits `RESOURCE_EXHAUSTED`. |
| 2.3 | Request finishing at the first token (max_new_tokens=1) and immediate EOS (`ignore_eos=false` on a prompt that stops at once) through P→D | **ADD** | Transfer for a request that is already complete; the bootstrap room must still be released. In `params_matrix.py` (`max_new_tokens_1`) + tiny prompts. |
| 2.4 | `max_new_tokens=0`, `max_new_tokens` > remaining context, ISL+OSL > context_length (prefill-ok/decode-prealloc-fails path) | PART | Plan has the generic boundary. Add the asymmetric case: ISL = context−100, max_new_tokens 1024 → clean 400 at admission, not a decode-side hang. `verify_pd_disagg.py --boundary` + `params_matrix.py`. |
| 2.5 | Same `rid` twice; duplicate content concurrently; client-supplied `bootstrap_host/room` fields | PART | Plan lists rid collision; add the client-supplied bootstrap fields (must be ignored/overwritten by the LB). |
| 2.6 | Client abort during chunked prefill, during transfer, during decode; abort storm (50 in 10 s) | HAVE (plan) / tool **ADD** | `abort_test.py` (being written): closes the stream, polls both engines until `token_usage` returns to baseline, then re-sends the same prompt and checks `cached_tokens`. |
| 2.7 | Resource return-to-baseline after **every** phase (not only after the soak) | **ADD** | Make it an assertion in every runner: both engines `num_running_reqs==0`, `token_usage<0.02` within 60 s of the last response. |

## 3. Sliding-window (gpt-oss) and page/chunk boundaries

| # | Case | Status | How |
|---|---|---|---|
| 3.1 | Prompts shorter than the window and around it: 1, 2, 63, 64, 65, 127, 128, 129, 191, 192, 193 tokens (page 64, window 128) | **ADD** | `tiny_prompts.py` (being written). The SWA pool uses "window + 1 page" blocks; off-by-one at window/page edges is the classic bug. Decode 160 tokens so generation crosses the window. |
| 3.2 | Chunk boundaries at **8192** (this build's chunk), not 32k as in the plan: 8191/8192/8193, 16383/16384/16385 + needle at 8192±64 | **ADD** | Same tool; plan's 32k boundary cases remain valid as "multiple of chunk". |
| 3.3 | Unaligned long prompts 127,999 / 255,123 / 511,777 / 999,999 | HAVE | prompt sets built. |
| 3.4 | Coherence past 128 generated tokens (window KV correctly transferred / zeroed on decode) | HAVE | 1,024-token outputs + 4-gram check + differential. |
| 3.5 | Prefix-cache hit ending mid-window; SWA-aware radix fallback to a shorter hit | HAVE (plan) | TC-2 edge list; add page-unaligned shared prefix 131,000 with page 64. |
| 3.6 | SWA pool sizing: `swa_full_tokens_ratio` default 0.8, `--disable-hybrid-swa-memory` variant | **ADD (variant)** | One run of TC-1 short sanity with hybrid SWA memory disabled: isolates SWA-pool bugs from attention bugs. |

## 4. Sampling / request-parameter plumbing across P→D

Penalisers, stop handling and logprobs are split between prefill (first token) and decode (rest). The plan only mentions `n>1`, `return_logprob`, `stop`, `logit_bias`.

| # | Case | Status |
|---|---|---|
| 4.1 | repetition / frequency / presence penalties (penalty state must include the prefill-produced first token) | **ADD** (`params_matrix.py`) |
| 4.2 | `min_new_tokens`, `stop_token_ids`, `stop_regex`, `no_stop_trim`, `skip_special_tokens=false` | **ADD** |
| 4.3 | `return_logprob` with `logprob_start_len=0` (input logprobs come from prefill) and `top_logprobs_num=5` | **ADD** |
| 4.4 | `top_k=1` at temperature 0.7 ≡ greedy; `sampling_seed` determinism | **ADD** |
| 4.5 | Grammar-constrained (`json_schema`, `regex`, `ebnf`) with `--grammar-backend none` → clean 400, never a hang | **ADD** |
| 4.6 | Streaming vs non-streaming equality, `[DONE]` terminator, `usage` in the last chunk through both LBs | **ADD** (`chat_suite.py`) |

## 5. Chat / Harmony / multimodal paths

| # | Case | Status |
|---|---|---|
| 5.1 | gpt-oss `reasoning_effort` low/medium/high, reasoning_content vs content split, `<|return|>`/`<|call|>` stop positions with `ignore_eos=false` | PART → `chat_suite.py` |
| 5.2 | gpt-oss tool calling (function definitions) through PD; multi-turn with dropped reasoning | **ADD** (`chat_suite.py`) |
| 5.3 | Qwen3-VL: N images (1/4/16/64) × resolutions, image straddling the 8k chunk, image-only, text-only, corrupt image → 400, same-image/different-question cache hit, different-image/same-text must miss | HAVE (plan) / tool HAVE (`mm_request.py`) |
| 5.4 | Qwen3-VL MRoPE after the hand-off: decode re-processes the images itself (no mrope state crosses the wire) → long generations (1,024 tokens) vs golden is the detector; also compare `prompt_tokens` between engines (image token count must agree) | HAVE (differential) + **ADD** (1.7 accounting) |
| 5.5 | Vision-encoder compile only on first sight of a new resolution (29 patch-count buckets precompiled: 196…3364 patches) | HAVE (TC-5) — note the bucket list in the log; a resolution outside it is a recompile case to test explicitly. |
| 5.6 | **Qwen3-VL-30B-A3B decoder is eager** (finding F-2): TC-5 "zero decoder recompiles" is vacuous for Qwen; TC-8 perf gates for Qwen are meaningless until a compile policy exists | **FLAG** |

## 6. Load-balancer layer

| # | Case | Status |
|---|---|---|
| 6.1 | Run the short sanity + `chat_suite` through **both** LBs: Python mini-LB (what CI uses, launched by the recipe launcher) and the Rust router (`--pd-disaggregation`, `--worker-startup-timeout-secs`) | **ADD** |
| 6.2 | Error propagation: engine 4xx must surface as 4xx (not 200 + truncated stream); decode death mid-stream must end the stream with an error (known mini-LB bug #1115 — verify on main) | **ADD** |
| 6.3 | Rust router no-rediscovery after `--worker-startup-timeout-secs` (plan risk #8) | HAVE (plan TC-0/TC-7) |

## 7. Compile hygiene (TC-5) specifics for this build

| # | Case | Status |
|---|---|---|
| 7.1 | Startup recompile count as baseline (gpt-oss golden: see `compile_audit.sh`; Qwen golden: 64 dynamo recompiles during bucket capture) | HAVE |
| 7.2 | First `return_logprob` request after warm-up; first `n>1`; first spec request with odd batch (bs × draft tokens) | **ADD** |
| 7.3 | Decode batch sizes not in `tpu_compile_bs` (3, 5, 7 when bs list is [1,2,4,8]) | HAVE (plan) |
| 7.4 | Prefill remainder chunks (ISL mod 8192 ∈ {1, 63, 64, 65, 8191}) | HAVE via unaligned + tiny prompts |
| 7.5 | `SGLANG_TPU_ALLOW_SPEC_RECOMPILE` unset in Config D: a verify-graph recompile crashes the decode (the CI tests set it) — run D without it first (that is the production stance), record the outcome | **ADD** |

## 8. Capacity / timeouts (re-derived for this build)

- KV ≈ 15.2 KB/token/rank for gpt-oss fp8 (incl. SWA pool) → 1M request ≈ 16 GB/rank, ≈ 120 GB over the wire; measured 102–128 Gbit/s → 10–30 s transfer. `SGLANG_RAIDEN_TIMEOUT_S=600` on both engines is ample; the default 120 s would also pass but leave little margin under contention.
- Pool bounds for gpt-oss: ≥ ~605k (precompile hang #363) and ≤ ~1.5M (OOM). Recipes use 786,432 (128k/256k) and 1,114,112 (512k/1M).
- Cold start measured: gpt-oss 389 s, Qwen 474 s (weights 99–131 s, decode graph capture 234–358 s).

## 9. What this adds to the execution queue

1. (running) determinism baseline + GSM8K-100 on golden gpt-oss.
2. Config A gpt-oss 128k: TC-0/TC-1 + `tiny_prompts` SWA sweep + `params_matrix` + `chat_suite` (both LBs) + `abort_test` + retraction test (MR 8) + KV-checksum variant.
3. Same for Qwen3-VL (eager decoder noted), with `mm_request` image matrix and accounting check.
4. `1m`-shape goldens (256k/512k/1M) on both hosts, then Config A `1m`.
5. B/C/D/E as planned, each with the resource-return assertion and KV canary on during eviction/overflow passes.
