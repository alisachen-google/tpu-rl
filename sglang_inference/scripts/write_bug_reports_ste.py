#!/usr/bin/env python3
"""Write the bug reports in ASD-STE100 style (short sentences, active voice, one instruction per step)
into results/bug_reports/. Run after each change to the findings. Idempotent."""
import os

OUT = os.path.expanduser("~/pd_disagg/results/bug_reports")
os.makedirs(OUT, exist_ok=True)

ENV = """## Test environment

| Item | Value |
|---|---|
| Software | sglang-torchtpu, branch `main`, commit `09d1244c26` (2026-10-07). sglang 0.5.13.dev7249 |
| TPU stack | torch-tpu 0.1.1+release.2026.9.22.20260927223741, libtpu 0.0.47, tpu-sync-torch 0.0.1.dev20261002213151, torch 2.11.0+cpu, transformers 5.8.1 |
| Hardware | Two TPU v7x hosts (4 chips, 8 TensorCores, 944 GiB RAM each). One host runs the prefill engine. One host runs the decode engine |
| Topology | Tensor parallel 8, expert parallel 8 on each host. KV transfer with the `raiden` backend. Python mini load balancer (mini-LB) on port 8080 |
| Recipes | `sglang_inference/recipes/*.json`. Start a pair with `scripts/pd.sh start <recipe> <run-name>` |
| Models | `openai/gpt-oss-120b` (MXFP4 weights, FP8 KV cache, page size 64). `Qwen/Qwen3-VL-30B-A3B-Instruct` (BF16 weights, FP8 KV cache, page size 128) |

All file paths in this report are relative to `sglang_inference/` in this repository.
"""

HOW_TO_START = """### Before you start

1. Make sure that no other user holds the TPU: `bash scripts/tpu_free.sh`.
2. Start the Config A pair for the model: `bash scripts/pd.sh start recipes/{recipe} {run}`.
3. Wait until the command reports `LB /health_generate ready`. This takes 4 to 8 minutes.
4. Set `LB=http://10.128.15.217:8080` (the mini-LB) and `PREFILL=http://10.128.15.217:30000`.
"""

REPORTS = {}

REPORTS["F-1"] = f"""# F-1: The server stops with a hard abort when it cannot create the tier-2 compile cache directory

**Type:** bug (robustness). **Severity:** low-medium (blocks every second user on a shared host). **Status:** reproduced 2 of 2, in two forms.

## Summary

The TPU runtime keeps a tier-2 compile cache in `/dev/shm/torch_tpu_cache`. On a shared host, another user can own this directory with mode 755.
Then the new server cannot create its cache sub-directory. The launcher ignores this error. torch_tpu does not ignore it. torch_tpu stops the
process with `SIGABRT`. SGLang then reports a misleading error: `Rank 0 scheduler died during initialization (exit code: -6)`.

## Effect

Every server start by a second user on the host fails. The real cause is only visible in the TPU runtime log lines.

{ENV}
## How to reproduce

1. On the host, create the cache directory as another user: `sudo -u <other-user> mkdir -p /dev/shm/torch_tpu_cache` and `sudo chmod 755 /dev/shm/torch_tpu_cache`.
2. Start any server with `--enable-torch-compile`. Example: `bash scripts/pd.sh golden recipes/qwen3-vl_G_128k_fp8_e4m3.json f1-repro b`.
3. Wait about 10 seconds after the line `Load weight begin`.

**Observed result.** The scheduler processes stop. The log shows, once per rank:
```
F0000 ... tier2_compilation_cache.cc:241] Check failed: status.ok() Failed to create tier-2 cache directory:
/dev/shm/torch_tpu_cache/<name>/<hash> with error: ...
RuntimeError: Rank 0 scheduler died during initialization (exit code: -6)
```
**Expected result.** The server starts without the tier-2 cache, or it stops with a clear message that names the directory and the permission problem.

## Second form of the same problem (2026-10-08)

When the launcher runs with `ENABLE_COMPILE_CACHE=0`, it sets no tier-2 name. torch_tpu then uses the name `default`, so every user's server writes to
`/dev/shm/torch_tpu_cache/default/<hash>/`. The first user owns that directory. The second user's server aborts on all ranks:
```
F0000 ... tier2_compilation_cache.cc:266] Check failed: lock_fd_.valid() Failed to create tier-2 cache entry lock file:
/dev/shm/torch_tpu_cache/default/ee8e2de63f954d59/<entry>.lock with error: Permission denied
```
So "compile cache off" is not a work-around, and two users with default settings cannot share a host.

## Work-around

Run `sudo chmod 1777 /dev/shm/torch_tpu_cache` on each host and give each user or job its own tier-2 name
(`TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=<unique name>`). Setting the variable to null does not help, because torch_tpu falls back to `default`.

## Possible cause and suggested fix

`benchmark/tpu/launch_server.py`, function `prepare_runtime_env`, creates the directory inside `try/except OSError: pass`. torch_tpu then treats
the same failure as fatal (`CHECK`). Suggested fix: when the directory cannot be created, the launcher must unset
`TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE` or stop with a clear error. torch_tpu should fall back to no tier-2 cache instead of `CHECK`.

## Related open issues in sgl-project/sglang-torchtpu

None found among the 69 open issues (checked 2026-10-08).

## Evidence

`results/logs/runs/G_qwen_128k_fp8/golden/server.attempt1_tier2cache_abort.log.gz`
"""

REPORTS["F-2"] = f"""# F-2: The Qwen3-VL-30B-A3B language model does not get compiled on TPU

**Type:** bug (missing compile policy, performance). **Severity:** high for performance tests. **Status:** reproduced on every start. Fix validated on hardware 2026-10-08: with the 14-line patch the decoder compiles, decode speed goes from 6 to 180 tokens/s, 0 serving-time recompiles; outputs differ numerically from eager; GSM8K-100 compiled 0.99 vs eager 0.97, so accuracy is unaffected.

## Summary

With `--enable-torch-compile`, the TPU graph runner compiles only the vision encoder of `Qwen3-VL-30B-A3B-Instruct`. The MoE language model runs
in eager mode. Decode speed is about 6 tokens per second for one request. gpt-oss-120b on the same stack decodes at 176 tokens per second.

## Effect

Every latency and throughput result for this model is invalid. The compile-hygiene test (TC-5) is empty for this model, because the decoder
never compiles and so it never recompiles.

{ENV}
## How to reproduce

1. Start the monolithic server: `bash scripts/pd.sh golden recipes/qwen3-vl_G_128k_fp8_e4m3.json f2-repro b`.
2. Wait for `/health` to return 200.
3. Send one request that generates 1,024 tokens:
   ```
   curl -s http://10.128.0.90:30000/generate -H 'Content-Type: application/json' \\
     -d '{{"text":"Write a long story about a lighthouse keeper.","sampling_params":{{"temperature":0,"max_new_tokens":1024,"ignore_eos":true}}}}'
   ```
4. Read the `Decode batch` lines in the server log.

**Observed result.** The request takes 160 to 195 seconds. The log shows `gen throughput (token/s): 6` to `13`. The startup log shows only
`VisionTpuCompilePolicy: installed qwen3_vl_vit_encoder` and no decoder policy.
**Expected result.** Decode speed in the same range as gpt-oss-120b (5 to 10 ms per step). The startup log shows a decoder compile policy.

## Fix and validation

Patch: `patches/F-2_qwen3_vl_moe_decoder_compile_policy.patch` registers the existing `Qwen3VLDecoderTpuCompilePolicy` for `Qwen3MoeLLMModel`
(mirrors `qwen3_vl.py` for the dense class; the same policy already serves the MoE Qwen3.5 decoder). Result on hardware (`results/F2_patch/`):

| Measure | Eager (main) | With patch |
|---|---|---|
| Decoder policy in log | none | `Installed Qwen3VLDecoderTpuCompilePolicy on Qwen3MoeLLMModel.forward` |
| Decode speed, 1 request | 6 to 13 tokens/s | 178 to 183 tokens/s |
| 1,024-token answer | 160 to 195 s | 5.9 to 7.1 s |
| Cold start | 474 s | about 1,160 s (decoder graph capture 1,094 s) |
| Compile artefacts written | small | 65 GB inductor + 51 GB tier-3 |
| Serving-time recompiles (35 prompt lengths) | 0 (nothing compiled) | 0 |
| Token identity with eager output | reference | differs after 9 to 83 tokens; first-token log-prob delta 0.03 to 0.25 |

The outputs differ because compiled and eager kernels are numerically different. Accuracy is unaffected: GSM8K (100 examples, greedy)
scores 0.99 compiled against 0.97 eager, in 29 s instead of about 17 minutes. Needle retrieval was correct in both modes. A warm start with the
caches from a previous launch takes 421 s.

## Possible cause and suggested fix

`python/sglang/srt/models/qwen3_vl.py`, lines 1380 to 1388, attach `Qwen3VLDecoderTpuCompilePolicy` only to the dense `Qwen3LLMModel`.
The MoE class `Qwen3MoeLLMModel` in `python/sglang/srt/models/qwen3_vl_moe.py` has no policy. `TpuGraphRunner`
(`python/sglang/srt/model_executor/tpu_graph_runner.py`, lines 216 to 295) skips the whole-model fallback because the vision policy was
installed. Suggested fix: register a decoder compile policy for `Qwen3MoeLLMModel`, or apply the whole-model fallback when only a vision policy
is installed.

## Related open issues in sgl-project/sglang-torchtpu

#1069 asks for a nightly Qwen3-VL-30B-A3B test (coverage request, no mention of the eager decoder). No issue reports the missing compile policy. New.

## Evidence

`results/logs/runs/G_qwen_128k_fp8/golden/server.log.gz` (search for `CompilePolicy`), `results/golden/qwen3-vl_fp8/run.log` (timings).
"""

REPORTS["F-3"] = f"""# F-3: Greedy decoding is not batch-invariant (test-method note, not a bug report)

**Type:** note for the test method. **Severity:** medium for test design. **Status:** measured on 10 of 10 prompts.

## Summary

The same monolithic gpt-oss-120b server gives different greedy outputs when the batch composition changes. We ran the same prompts once at
concurrency 2 and once at concurrency 1. All 10 outputs differ after 3 to 59 tokens. The first-token log-probability is identical in 4 of 5
prompt sets. When we repeat a run with the same concurrency, the outputs are identical.

## Effect

A "token-exact" comparison between the PD system and the monolithic system is valid only at the same concurrency (we use concurrency 1).
For concurrent runs, use these gates instead: the needle answers, the exact token count, the finish reason, and the GSM8K score.

{ENV}
## How to reproduce

1. Start the monolithic gpt-oss server: `bash scripts/pd.sh golden recipes/gpt-oss_G_128k_fp8_e4m3.json f3-repro a`.
2. Generate outputs at concurrency 2: `bash scripts/run_golden.sh gpt-oss http://10.128.15.217:30000 results/golden/c2 1024,8192,32768 2 1024`.
3. Generate the same outputs at concurrency 1: `bash scripts/run_golden.sh gpt-oss http://10.128.15.217:30000 results/golden/c1 1024,8192,32768 1 1024`.
4. Compare: `python scripts/golden_diff.py diff --a results/golden/c2/8192.jsonl --b results/golden/c1/8192.jsonl --report /tmp/r.json`.

**Observed result.** `identical=False`, token prefix 22 to 59, first-token log-probability delta 0.0000.
**Expected result (for a batch-invariant stack).** `identical=True`.

## Related open issues in sgl-project/sglang-torchtpu

#321 (greedy output differs cached vs cold) and #391 (different DP ranks give different results) describe other sources of greedy non-reproducibility on TPU. Our batch-composition effect is not listed. Related, not a duplicate.

## Evidence

`results/golden/gpt-oss_fp8_rep2_c1/determinism_*.json`, `results/golden/gpt-oss_fp8_rep2_c1/determinism.txt`.
"""

REPORTS["F-4"] = f"""# F-4: gpt-oss-120b in PD mode gives a different output than the monolithic server for long prompts

**Type:** numerical difference (not a PD defect). **Severity:** low / informational. **Status:** resolved 2026-10-08: with `--disable-hybrid-swa-memory` on both sides, PD and monolithic are token-identical for 6 of 6 prompts at 8k, 32k and 128k; GSM8K through PD 0.98 vs 0.99 monolithic.

## Summary

We compare the PD system (Config A: 1 prefill + 1 decode, raiden, radix cache off) with the monolithic server. Both run at concurrency 1 with
greedy sampling. For gpt-oss prompts of 32,768 tokens and more, the outputs differ from token 3 to 19. The first-token log-probability differs
by up to 0.16. The prefill engine computes the first token, so the difference is in the prefill forward pass.

The same test gives identical results in these cases:
- gpt-oss prompts of 1 to 16,385 tokens (37 prompts, including 2- and 3-chunk prompts): identical.
- Qwen3-VL-30B (no sliding-window layers) prompts of 1 to 131,072 tokens (43 prompts, up to 16 chunks): identical for all 1,024 output tokens.
- Monolithic servers with different context sizes (128k and 1M): identical.

| Model | Prompt tokens | Chunks | Identical output | First-token log-prob delta |
|---|---|---|---|---|
| gpt-oss | 1,024 and 8,192 | 1 | yes (4 of 4) | 0.0000 |
| gpt-oss | 8,193 and 16,385 (160 output tokens) | 2 to 3 | yes (2 of 2) | 0.0000 |
| gpt-oss | 32,768 | 4 | no (prefix 3 tokens) | 0.104, 0.156 |
| gpt-oss | 131,072 | 16 | no (prefix 5, 15 tokens) | 0.032, 0.031 |
| Qwen3-VL | 1,024 to 131,072 | 1 to 16 | yes (8 of 8) | 0.0000 |

## Cause (high confidence)

The prefill chunk sizes differ between the two modes for gpt-oss only. For the same 131,072-token prompts, the PD prefill engine uses chunks of
8192, 2752, 2880 and 8064 tokens. The monolithic server uses chunks of 8064, 1344 and 8192 tokens. Qwen uses 8192 in both modes. The chunk size
is limited by the free slots in the small hybrid sliding-window (SWA) pool (6,144 tokens in this configuration). The two modes use that pool
differently, because the PD prefill engine holds SWA pages for the KV transfer. Different chunk boundaries change the accumulation order in
bf16 and FP8 arithmetic. After a few tokens the greedy argmax changes.

So the difference is numerical, not a KV-transfer fault. The needle answers are correct on both sides.

## Effect

Outputs from the PD system and the monolithic system are not interchangeable for long gpt-oss prompts. The effect on task accuracy is not
measured yet (GSM8K through PD is queued).

{ENV}
## How to reproduce

1. Build the prompt sets one time: `python scripts/build_prompts.py --model-path <gpt-oss snapshot> --out prompts/gpt-oss --lengths 1024,8192,32768,131072 --per-length 2`.
2. Start the monolithic server: `bash scripts/pd.sh golden recipes/gpt-oss_G_128k_fp8_e4m3.json f4-golden a`.
3. Generate the golden outputs at concurrency 1: `bash scripts/run_golden.sh gpt-oss http://10.128.15.217:30000 results/golden/f4 1024,8192,32768,131072 1 1024`.
4. Stop the server: `bash scripts/pd.sh stop`.
5. Start the PD pair: `bash scripts/pd.sh start recipes/gpt-oss_A_128k.json f4-pd`.
6. Generate the PD outputs through the mini-LB at concurrency 1: `bash scripts/run_tc1.sh gpt-oss http://10.128.15.217:8080 results/f4-pd 1024,8192,32768,131072 results/golden/f4 1`.
7. Read the `diff vs golden` tables in `results/f4-pd/tc1.log`.
8. Optional: compare the chunk sizes. Run `grep -oE '#new-token: [0-9]+' <server.log> | sort | uniq -c` on the prefill log and on the monolithic log.

**Observed result.** `identical=True` for 1,024 and 8,192 tokens. `identical=False` with a token prefix of 3 to 15 for 32,768 and 131,072 tokens.
The chunk-size histograms differ.
**Expected result.** `identical=True` for all lengths, or a documented statement that PD and monolithic outputs differ for hybrid-SWA models.

## Confirmation (done)

1. Config A and the golden with `--disable-hybrid-swa-memory` (`recipes/gpt-oss_A_128k_noswamem.json`, `recipes/gpt-oss_G_128k_fp8_e4m3_noswamem.json`): both modes use 8192-token chunks and the outputs are **identical for 6 of 6 prompts** (8k, 32k, 128k; 1,024 tokens each). Control: the monolithic server with SWA memory on vs off differs at 32k, so the SWA-memory mode itself changes the numerics.
2. GSM8K (100 examples) through PD: **0.98**; monolithic: 0.99. No measurable accuracy effect.

## Suggested fix

Make the chunk schedule independent of the serving mode, for example by giving the PD prefill engine the same SWA-pool budget per chunk as the
monolithic server. Or document the difference.

## Related open issues in sgl-project/sglang-torchtpu

#321 is the closest (cached vs cold greedy difference on one host). Our PD vs monolithic difference and its chunk-schedule cause are not reported. New.

## Evidence

`results/A/tc1/gpt-oss/golden_diff_c1_*.json`, `results/A/tc1/gpt-oss/rep2/pd_self_*.json`, `results/A/tc1/qwen3-vl/golden_diff_c1_*.json`,
`results/A/tiny/*/tiny.jsonl` versus `results/golden/*_1m_c1/tiny.jsonl`, chunk histograms in `results/FINDINGS.md`.
"""

REPORTS["F-5"] = f"""# F-5: One request that asks for prompt log-probabilities stops the whole prefill engine

**Type:** bug (crash). **Severity:** high. **Status:** reproduced 2 of 2 (2026-10-07 and 2026-10-08).

## Summary

A client sends a normal `/generate` request with `return_logprob=true` and `logprob_start_len=0` (prompt log-probabilities). The TPU compiled
backend does not support prompt log-probabilities unless `SGLANG_TPU_ENABLE_PROMPT_LOGPROB=1` is set. The backend rejects the request inside the
model forward pass with `NotImplementedError`. At that point the only possible result is a scheduler crash. All 8 prefill scheduler ranks stop.
The launcher then stops the load balancer. Every later request gets HTTP 500.

## Effect

Any client can stop a prefill engine with one request. In a large deployment this takes down the prefill engines one by one.

{ENV}
## How to reproduce

{HOW_TO_START.format(recipe="gpt-oss_A_128k.json", run="f5-repro")}
### Steps

1. Send one request that asks for prompt log-probabilities:
   ```
   curl -s -m 120 $LB/generate -H 'Content-Type: application/json' \\
     -d '{{"text":"hi there","sampling_params":{{"max_new_tokens":4}},"return_logprob":true,"logprob_start_len":0}}'
   ```
2. Wait 15 seconds.
3. Check the prefill engine: `curl -s -o /dev/null -w '%{{http_code}}\\n' $PREFILL/health`.

**Observed result.** Step 1 returns `Internal Server Error` (HTTP 500). Step 3 returns `000` (no connection). The prefill log shows, once per rank:
```
[TPx] Scheduler hit an exception: Traceback (most recent call last):
  File ".../sglang/srt/disaggregation/prefill.py", line 773, in event_loop_overlap_disagg_prefill
  ...
  File ".../sglang/srt/hardware_backend/tpu/compilation/compiled_forward_shim.py", line 967, in _forward
    raise NotImplementedError(rejection)
NotImplementedError: input/prompt logprobs (echo) need SGLANG_TPU_ENABLE_PROMPT_LOGPROB=1 on the TPU compiled backend;
only output (next-token) logprobs are served by default.
```
**Expected result.** Step 1 returns HTTP 400 with the same message. Step 3 returns 200.

## Possible cause and suggested fix

The check is in the model forward pass (`compiled_forward_shim.py:967`). Move the check to request validation in the tokenizer manager or the
HTTP layer, and return HTTP 400. Or decide the option one time at server start.

## Also to test

The same request against the monolithic server. We expect the same crash, because the check is not specific to PD mode.

## Related open issues in sgl-project/sglang-torchtpu

**#432** (2026-08-19, open, labelled feature request) points at the same code (`compiled_forward_shim.py`): prompt logprobs are not supported on TPU. It does not state that one request crashes all prefill schedulers. Our report adds the crash and the PD impact; link the two.

## Evidence

`results/logs/runs/A_gptoss_128k_run1/prefill/server.attempt1_crash_prompt_logprob.log.gz`, `results/A/params/gpt-oss/params_pd_A.json`.
"""

REPORTS["F-6"] = f"""# F-6: A request with `n=2` gets no response

**Type:** bug (hang). **Severity:** high. **Status:** reproduced 3 of 3 (temperature 0 and 0.7). The decode engine keeps one running request with 128 used tokens until the client disconnects; both engines stay healthy.

## Summary

During the sampling-parameter matrix on Config A (gpt-oss), the case `n=2` (two parallel samples, temperature 0, 128 new tokens) did not return
within 300 seconds. All 9 cases before it returned in less than 2 seconds. A later crash (F-5) stopped the engine before we could inspect the
request. The repeat has not run yet.

{ENV}
## How to reproduce

{HOW_TO_START.format(recipe="gpt-oss_A_128k.json", run="f6-repro")}
### Steps

1. Start the watchdog in a second terminal: `python scripts/pd_watchdog.py --server prefill=$PREFILL --server decode=http://10.128.0.90:30000 --out /tmp/wd.csv`.
2. Send the request:
   ```
   curl -s -m 300 $LB/generate -H 'Content-Type: application/json' \\
     -d '{{"text":"The three primary colors are","sampling_params":{{"temperature":0,"max_new_tokens":32,"n":2}}}}' -w '\\nHTTP %{{http_code}} in %{{time_total}}s\\n'
   ```
3. If the request does not return in 300 seconds, read the watchdog output for `num_running_reqs`, `num_decode_prealloc_queue_reqs` and `num_decode_transfer_queue_reqs`.

**Observed result (first run).** No response in 300 seconds.
**Expected result.** Two outputs within a few seconds, or HTTP 400 if `n>1` is not supported in PD mode.

## Cause (located in the code)

The tokenizer manager expands one `n=2` request into a prefix request (`max_new_tokens 0`) plus two sampling requests, all built from the
same prompt object, so all three carry the same `bootstrap_room` (the second room produced by `_normalize_bootstrap_params` is never used).
Three requests with one room collide in the transfer bookkeeping; the decode engine keeps one of them with the prompt pages allocated and
nothing to generate, and no deadline fires while its receiver waits. Fix: index bootstrap rooms per sub-request and do not send the
prefix-only request to the decode in disaggregated mode. Full trace: `F-6_github_issue.md`.

## Related open issues in sgl-project/sglang-torchtpu

None found. New.

## Evidence

`results/A/params/gpt-oss/params_pd_A.json` (case `n_2`, `HANG?`); repeat 2026-10-08 02:52 UTC in `results/phase_A_gptoss_redo.log` (3 of 3, decode engine holds one running request with 128 used tokens until the client disconnects).
"""

REPORTS["F-7"] = f"""# F-7: After the prefill engine dies, the decode engine recovers in about two minutes (observation, not a bug report)

**Type:** observation. **Severity:** low. **Status:** observed one time.

## Summary

When the prefill engine crashed (F-5), one request was in transfer on the decode engine. The decode metrics showed `num_running_reqs=1`,
`num_decode_transfer_queue_reqs=1` and a growing `kv_transfer_oldest_inflight_age_seconds`. All ranks logged
`Lost connection with prefill instance (bootstrap_addr: 10.128.15.217:8998), 0 requests affected`. About two minutes later, the request was
dropped (`num_transfer_failed_reqs_total` went from 1 to 2) and all gauges returned to zero. The decode engine stayed healthy.

The log line says `0 requests affected`, but one request was affected. This message is misleading.

{ENV}
## How to reproduce

1. Start Config A (gpt-oss) as described in F-5.
2. Start one long request (131,072 tokens, 1,024 new tokens) through the mini-LB.
3. While the request runs, stop the prefill engine on host a: `pkill -9 -f sglang.launch_server` (as the owner of the process).
4. Poll the decode metrics every 60 seconds: `curl -s http://10.128.0.90:30000/metrics | grep -E 'num_running_reqs|num_decode_transfer_queue_reqs|num_transfer_failed_reqs_total'`.

**Observed result.** Gauges return to zero within about 2 minutes. The `Lost connection` line reports `0 requests affected`.
**Expected result.** The same recovery. The log line should report the real number of affected requests.

## Related open issues in sgl-project/sglang-torchtpu

**#1113** (2026-10-07, P0) describes the same mechanism: when the prefill side fails, the decode keeps the request until the raiden deadline (120 s). Our ~2-minute recovery after a prefill crash matches it. Duplicate in mechanism; our data point confirms it for a full prefill-engine death.

## Evidence

`results/A/tc7_prefill_death_decode_state.log`, `results/logs/runs/A_gptoss_128k_run1/decode/server.log.gz`.
"""

REPORTS["F-8"] = f"""# F-8: A client that disconnects during chunked prefill stops the whole prefill engine

**Type:** bug (crash). **Severity:** high. **Status:** reproduced 2 of 2. Aborts during the decode phase (6, two concurrent) pass: resources return to baseline.

## Summary

A client sends a long prompt (131,072 tokens) as a streaming request. It closes the connection 2 seconds later. At that time the prefill
engine still processes the prompt in chunks. The prefill engine handles the abort in `_drain_deferred_chunked_sender_aborts` and calls
`release_kv_cache` for the request. The KV cache of the request is not fully allocated yet. The tree cache (`ChunkCache`, because the radix cache
is disabled in this configuration) does not permit this and stops with an `AssertionError`. All 8 prefill scheduler ranks stop. The launcher
stops the load balancer.

## Effect

A client disconnect is a normal event. In this build it stops the prefill engine and ends every in-flight request.

{ENV}
## How to reproduce

{HOW_TO_START.format(recipe="gpt-oss_A_128k.json", run="f8-repro")}
### Steps

1. Make sure that the prompt set exists: `prompts/gpt-oss/131072.jsonl` (see `scripts/build_prompts.py`).
2. Run the abort tool once, in the prefill phase:
   ```
   python scripts/abort_test.py --url $LB --prompt-file prompts/gpt-oss/131072.jsonl --phase prefill --abort-after 2 --repeat 1 \\
     --server-info $PREFILL --server-info http://10.128.0.90:30000 --settle 30
   ```
   The tool opens a streaming request, waits 2 seconds, and closes the connection.
3. Wait 15 seconds.
4. Check the prefill engine: `curl -s -o /dev/null -w '%{{http_code}}\\n' $PREFILL/health`.

**Observed result.** Step 4 returns `000`. The prefill log shows, once per rank:
```
[TPx] Scheduler hit an exception: Traceback (most recent call last):
  File ".../sglang/srt/disaggregation/prefill.py", line 672, in get_next_disagg_prefill_batch_to_run
    self.process_pending_chunked_abort()
  File ".../sglang/srt/managers/scheduler.py", line 3795, in _drain_deferred_chunked_sender_aborts
    release_kv_cache(req, self.tree_cache, is_insert=False)
  File ".../sglang/srt/mem_cache/common.py", line 298, in release_kv_cache
    assert tree_cache.supports_mamba(), (
AssertionError: Only a mamba-capable tree cache allows freeing before alloc
```
The tokenizer manager also logs `Received output for rid='abort-0-...' but the state was deleted in TokenizerManager`.
**Expected result.** Step 4 returns 200. The engine stops launching chunks for the request, releases its pages once after the raiden transfer group has drained, leaves every other request untouched, returns to baseline usage within 60 seconds, and serves the next request. Decode-phase disconnects already behave this way.

## Cause (located in the code) and suggested fix

The KV row of the aborted request is released twice. `process_pending_chunked_abort` defers the release because the raiden sender must drain
first, and parks the request in a deferred list. One step later the result of the chunk that was already in flight reaches
`_retire_aborted_prefill_result`, which sees that the request still holds its row and frees it. When the deferred list is drained,
`release_kv_cache` is called again without a `holds_kv` guard, and the routine asserts because only a mamba-capable cache may release without
a row. Fix: let the deferred list own the release (skip the free in `_retire_aborted_prefill_result` while the sender is not drained) and guard
the drain with `holds_kv`. Full trace with line references: `F-8_github_issue.md`.

## Control: radix cache on

Config B (same flags, radix cache on) ran the same prefill-phase abort on 2026-10-08: no crash, both engines healthy, follow-up requests
correct. The crash needs `--disable-radix-cache` (chunk cache).

## Related open issues in sgl-project/sglang-torchtpu

None found. #1114 covers a different hold (prefill keeps a request after decode gives up). New.

## Evidence

`results/logs/runs/A_gptoss_128k_run1/prefill_server.attempt2_abort_crash.log.gz`, `results/A/abort/gpt-oss/abort_prefill.json`.
"""

REPORTS["F-9"] = f"""# F-9: Prompt lengths that are not a multiple of the page or chunk size give a different first token in PD mode (resolved: not a bug)

**Type:** bug candidate (correctness). **Severity:** none. **Status:** resolved 2026-10-08. Against a concurrency-1 golden, two 127,999-token Qwen3-VL prompts are token-identical for all 1,024 output tokens (delta 0.0000). The earlier deltas came from a concurrency-8 golden (F-3 batch effect).

## Summary

For Qwen3-VL-30B, the PD system and the monolithic server give bit-identical first-token log-probabilities for every prompt whose length is a
multiple of the page size (128) and the chunk size (8,192): 1,024, 8,192, 32,768 and 131,072 tokens. For two prompts of 127,999 tokens the
values differ (delta 0.011 and 0.228). gpt-oss shows the same pattern for its unaligned prompts (delta 0.0004 and 0.005), in addition to the
larger effect in F-4.

## Effect

Most real prompts are not aligned. If this difference is real, almost every long request in PD mode differs from the monolithic result.

{ENV}
## How to reproduce

1. Build the prompt set: `python scripts/build_prompts.py --model-path <Qwen3-VL-30B snapshot> --out prompts/qwen3-vl --lengths 131072 --unaligned 127999 --per-length 2`.
2. Start the monolithic Qwen server: `bash scripts/pd.sh golden recipes/qwen3-vl_G_128k_fp8_e4m3.json f9-golden b`.
3. Generate the goldens at concurrency 1: `bash scripts/run_golden.sh qwen3-vl http://10.128.0.90:30000 results/golden/f9 131072,127999 1 64`.
4. Stop the server: `bash scripts/pd.sh stop`.
5. Start the PD pair: `bash scripts/pd.sh start recipes/qwen3-vl_A_128k.json f9-pd`.
6. Generate the PD outputs: `bash scripts/run_tc1.sh qwen3-vl http://10.128.15.217:8080 results/f9-pd 131072,127999 results/golden/f9 1`.
7. Compare the `first_token_logprob` field for each prompt id in `results/golden/f9/*.jsonl` and `results/f9-pd/*.jsonl`.

**Observed result (first run, concurrency-8 golden).** Delta 0.0 for 131,072 tokens. Delta 0.011 and 0.228 for 127,999 tokens.
**Observed result (2026-10-08, concurrency-1 golden).** Delta 0.0000 and identical 1,024-token outputs for both 127,999-token prompts (`results/A/tc1/qwen3-vl/golden_diff_c1_127999.json`).
**Expected result.** Delta 0.0 for all lengths. Met.

## Cause

The golden used in the first run was generated at concurrency 8. Greedy decoding on this backend is not batch-invariant (F-3), so the
comparison measured batch composition, not alignment. PD prefill handles the partial last chunk and page correctly.

## Related open issues in sgl-project/sglang-torchtpu

None found. New (still unconfirmed).

## Evidence

`results/phase_A_qwen.log` (section `first-token logprob PD vs golden`), `results/A/tc1/qwen3-vl/127999.jsonl`, `results/golden/qwen3-vl_fp8/127999.jsonl`.
"""

REPORTS["F-10"] = f"""# F-10: One large image (1920×1080) stops the Qwen3-VL prefill engine

**Type:** bug (crash). **Severity:** high. **Status:** reproduced 2 of 2 (2026-10-07 and 2026-10-08). 896×504 works.

## Summary

A client sends one chat request with one 1920×1080 PNG image. The vision encoder (ViT) on TPU has precompiled shapes for up to 4,900 patches.
The image has 8,160 patches. The ViT runner logs that it will do a one-time recompile. But after warm-up the server froze torch.compile with the
stance `fail_on_recompile`. The recompile raises `RuntimeError`. All 8 prefill scheduler ranks stop. The launcher stops the load balancer.
Images of 448×448 (1, 4 and 16 images) and 1024×1024 work.

## Effect

Any client image larger than the precompiled range stops the prefill engine. The default configuration has no `max_pixels` limit, so the
server accepts the image.

{ENV}
## How to reproduce

{HOW_TO_START.format(recipe="qwen3-vl_A_128k.json", run="f10-repro")}
### Steps

1. Send one request with one 1920×1080 image:
   ```
   python scripts/mm_request.py --url $LB --model qwen3-vl-30b --num-images 1 --resolution 1920x1080 --max-tokens 48 --temperature 0
   ```
   The tool draws a synthetic PNG and sends it as a `data:` URL in a `/v1/chat/completions` request.
2. Wait 20 seconds.
3. Check the prefill engine: `curl -s -o /dev/null -w '%{{http_code}}\\n' $PREFILL/health`.

**Observed result.** Step 1 returns HTTP 500 after about 65 seconds. Step 3 returns `000`. The prefill log shows:
```
[TPx] ViTTpuCompileRunner: patch count 8160 exceeds the largest precompiled bucket 4900; this image triggers a one-time ViT recompile.
      Raise SGLANG_TPU_VIT_MAX_PRECOMPILE_PATCHES, or lower max_pixels via --mm-process-config.
[TPx] Scheduler hit an exception: Traceback (most recent call last):
  File ".../sglang/srt/models/qwen3_vl.py", line 1801, in forward
  File ".../sglang/srt/managers/mm_schedule.py", line 354, in _encode_vision_items
  ...
RuntimeError: Detected recompile when torch.compile stance is 'fail_on_recompile'.
```
**Expected result.** The server compiles the new shape one time and answers. Or it returns HTTP 400 or 413 for an image that is too large.

## Cause (located in the code) and suggested fix

The generic ViT runner used by Qwen3-VL sends an image above the top precompiled bucket to the compiled function, which recompiles; the
global `fail_on_recompile` stance (armed at `Warmup complete; froze torch.compile`) turns the recompile into a `RuntimeError`. The runner
keeps the uncompiled encoder as `eager_fn` but never calls it. The MoonViT and Gemma4 vision policies in the same directory run such images
through `eager_fn` instead. Fix: add the same fallback to `VisionTpuCompilePolicy.run` (or the Qwen3-VL run adapter); or reject images above the
precompiled range with HTTP 413; or ship a `max_pixels` default that matches the compiled range (the Qwen3-VL-8B recipe uses 313,600). In all
cases catch encoder exceptions per request so the scheduler survives. Full trace: `F-10_github_issue.md`.

## Also to test

Other non-square sizes, for example 896×504. A request with `--mm-process-config '{{"image":{{"max_pixels":313600}}}}'` should downscale the image and work.

## Related open issues in sgl-project/sglang-torchtpu

**#488** (2026-08-25) reports the same mechanism with a different trigger: a constrained-decoding request causes a recompile, the `fail_on_recompile` stance raises, the scheduler exits. Ours is the first report for the vision encoder with a large image. #1068 proposes a recompile test gate. Same root cause class; link to #488.

## Evidence

`results/logs/runs/A_qwen_128k_run1/prefill_server.log.gz` (from line 7648), `results/A/mm/qwen3-vl/img1_1920x1080.json`.
"""

REPORTS["F-11"] = f"""# F-11: Chunked prefill stops forever when the SWA continuation budget reaches 0 before window eviction runs (gpt-oss, `max-running-requests` 1, TPU)

**Type:** bug (hang, deadlock). **Severity:** high. **Status:** reproduced 4 of 4 in the 1M-context recipe. Root cause located in the code. Not reproduced with 4 request slots (0 of 6 hang).

## Summary

gpt-oss-120b has sliding-window attention layers (window 128). In the disaggregated prefill recipe the radix cache is off and
`--max-running-requests 1` is set. The sliding-window (SWA) KV pool is then sized from a worst-case formula: two prefill chunks plus a small
per-slot allowance plus one page. A TPU-only rule rounds the pool down to a multiple of 8 pages. With one slot the rounding removes all the
headroom: the pool is exactly two chunks (16,384 tokens). The scheduler admits a continuation chunk only if window plus one page (192 tokens)
stay free. After two chunks only 192 tokens are free, so the third chunk gets size 0 and is never scheduled. Pages that left the window are
freed only when the next chunk of the request runs. Nothing runs, nothing is freed, the request waits forever. Both engines report healthy.

| Request slots | Pool from the formula | Pool after TPU rounding | Headroom above two chunks | Result |
|---|---|---|---|---|
| 1 | 16,832 | 16,384 | 0 | hang on every prompt longer than 16,384 tokens |
| 4 | 17,984 | 17,920 | 1,536 | works; pool cycles 0.54 to 0.91 |
| 8 | 19,520 | 19,456 | 3,072 | works |

## Effect

The 1M-context configuration (one request slot) cannot serve any prompt above about 16k tokens. A stalled request holds the only slot until
the client disconnects. No watchdog fires because no forward pass is in flight.

{ENV}
## How to reproduce

{HOW_TO_START.format(recipe="gpt-oss_A_1m.json", run="f11-repro")}
### Steps

1. Confirm the pool size in the prefill log on host a: `grep 'aligning swa_max_total_num_tokens' results/runs/f11-repro/prefill/server.log`. Expect `16832 -> 16384`.
2. Send one 32,768-token prompt: `python scripts/verify_pd_disagg.py --url $LB --router --lengths 32768 --timeout 600`.
3. After 60 seconds read the prefill metrics: `curl -s $PREFILL/metrics | grep -E 'swa_token_usage|num_running_reqs'`.
4. Read the prefill log: `grep 'Prefill batch' results/runs/f11-repro/prefill/server.log | tail -3`.

**Observed result.** Step 2 gets no answer in 600 seconds. Step 3 shows `swa_token_usage 0.99` and `num_running_reqs 1`. Step 4 shows two
batches, `#new-token: 8192` then `#new-token: 8000`, and nothing after. The request stays until the client disconnects.
**Expected result.** The 32k prompt answers in a few seconds, as it does with `--max-running-requests 4`.

## Cause (located in the code, commit 09d1244c26)

1. `model_executor/pool_configurator.py`, `SWAChunkCapPoolConfigurator` and `compute_swa_request_cap`: pool = 2 x chunked_prefill_size + (window + eviction_interval + 2 pages) x slots + 1 page. With one slot: 16,832 tokens.
2. `mem_cache/kv_cache_configurator.py`, TPU branch: rounds down to a multiple of 8 pages (512 tokens): 16,384. Log line `[hybrid-SWA] aligning swa_max_total_num_tokens 16832 -> 16384`.
3. `mem_cache/prefill_budget.py`, `_continuation_chunk_cap`: next chunk may use `remaining_swa - (window + page)`. Chunk 1 = 8,192, chunk 2 = 8,000, chunk 3 = 0 and `fit_chunk` returns None.
4. `managers/schedule_batch.py`, `maybe_evict_swa`: out-of-window SWA pages of a chunked request are freed only when a batch with that request runs (from the third chunk on with overlap scheduling). No batch, no eviction. Deadlock.

## Suggested fix

Round the TPU alignment up, or apply it before the headroom is added. Also evict the chunked request's out-of-window pages before its
continuation budget is computed, and fail the request with an error when no chunk fits and nothing else runs.

## Related open issues in sgl-project/sglang-torchtpu

None describe this. #1067 (long-context PD stress test for gpt-oss) would hit it with one request slot.

## Evidence

`results/F11/` (isolation runs `gpt-oss_A_1m_mr4_*`), `results/runs/A_gptoss_1m_run2/decode/server.log`, prefill log of `A_gptoss_1m_run2` on host a, `results/FINDINGS.md` (F-11 entries).
"""

REPORTS["F-12"] = f"""# F-12: The P/D handshake never completes and both engines hold the request with no deadline (candidate)

**Type:** bug (hang). **Severity:** high. **Status:** reproduced on 3 of 3 freshly started pairs with the `gpt-oss_A_128k_mr1` recipe (every request to those pairs stalls, including 8k single-chunk prompts). Trigger: `max-running-requests 1` on both engines is necessary (the same recipe with 2 or 8 slots serves everything) but not sufficient (the 1M-context single-slot recipe works); see `F-12_github_issue.md`, section "Trigger conditions". Pairs with the 1M-context recipe (same flags, larger context and pool) and with 8 request slots did not show it. Trigger not yet isolated.

## Summary

The first request to a new gpt-oss pair (8,192-token prompt, one chunk, `--max-running-requests 1`) was prefilled and then never produced a
token. The prefill engine kept the request in its inflight queue (one running request, SWA usage 0.5, scheduler idle). The decode engine never
ran a batch. Its scheduler polled the request's KV receiver in the handshake wait and the receiver stayed in the `Bootstrapping` state for 30
minutes. No error was logged on either side. When the client disconnected, both sides logged the abort at once (`Prefill transfer failed`,
`Decode handshake failed`, both `Aborted by AbortReq`). The next request stalled in the same way.

## Effect

One request occupies the only slot on both engines until the client gives up. No timeout clears it.

{ENV}
## How to reproduce (observed once, trigger not yet isolated)

{HOW_TO_START.format(recipe="gpt-oss_A_128k.json with --max-running-requests 1", run="f12-repro")}
### Steps

1. Send one 8,192-token prompt: `python scripts/golden_diff.py generate --prompts prompts/gpt-oss/8192.jsonl --url $LB --out /tmp/f12.jsonl --max-new-tokens 64 --concurrency 1 --timeout 900`.
2. After 60 seconds read both engines: `curl -s $PREFILL/metrics | grep -E 'num_running_reqs|inflight'` and `curl -s http://10.128.0.90:30000/metrics | grep -E 'num_running_reqs|prealloc|transfer'`.
3. Dump the decode scheduler: `py-spy dump --pid <decode TP0 scheduler pid> --nonblocking`.

**Observed result.** Prefill: one running request in the inflight queue. Decode: all queues 0, scheduler in `pop_preallocated -> _update_handshake_waiters -> poll_and_all_reduce`.
**Expected result.** 64 tokens within seconds, or a failed request within the 300 s bootstrap or waiting timeout.

## What the code says

The raiden sender moves its room to `WaitingForInput` without waiting for the decode and bounds the handshake with `fail_decode_silence`
(300 s). That deadline did not fire, so the prefill believed the decode had engaged. The raiden receiver's only deadline (`_check_waiting_timeout`,
300 s) starts in `send_metadata`, after the room leaves `Bootstrapping`. While the room is still bootstrapping, `poll()` has no deadline and the
decode prealloc queue keeps waiting. When the two sides disagree about the handshake state, nothing times out.

## Related open issues in sgl-project/sglang-torchtpu

#1113 and #1078 describe decode waits that at least expire after the transfer or reshard deadline. This wait does not expire.

## Evidence

`results/F11/handshake_stall/decode_tp0_pyspy.txt`, `results/F11/handshake_stall/prefill_tp0_pyspy.txt`, `results/logs/runs/F11_gpt-oss_A_128k_mr1/*.log.gz`, `results/F11/gpt-oss_A_128k_mr1_8192.log`. Second pair: `results/F12/` (client logs, decode stack dump), `results/runs/F12_gptoss_A_128k_mr1_try1/`.
"""

INDEX = """# Bug reports: sglang-torchtpu, 1 prefill + 1 decode on TPU v7x

The chronological raw log is `../FINDINGS.md`. These reports use ASD-STE100 style: short sentences, active voice, one instruction per step. Each report has a numbered
"How to reproduce" section with the exact commands. The tools are in `../../scripts/`. The server recipes are in `../../recipes/`.

| ID | Title | Type | Severity | Status | Overlap with open issues |
|---|---|---|---|---|---|
| [F-1](F-1.md) | Hard abort when the tier-2 compile cache directory cannot be created | bug (robustness) | low | reproduced, work-around  none |
| [F-2](F-2.md) | Qwen3-VL-30B-A3B language model is not compiled on TPU | bug (perf, missing policy) | high for perf tests | reproduced on every start  #1069 (coverage request only) |
| [F-3](F-3.md) | Greedy decoding is not batch-invariant | test-method note | medium | measured  #321/#391 related |
| [F-4](F-4.md) | gpt-oss PD output differs from monolithic for long prompts (chunk schedule differs) | numerical difference | high until accuracy check | reproduced 6/6, cause identified  #321 related |
| [F-5](F-5.md) | A prompt-logprob request stops the prefill engine | bug (crash) | high | reproduced 2/2  #432 (same code, no crash reported) |
| [F-6](F-6.md) | `n=2` request gets no response | bug (hang) | high | reproduced 3/3  none |
| [F-7](F-7.md) | Decode engine recovers about 2 min after the prefill engine dies | observation | low | observed  #1113 (same mechanism) |
| [F-8](F-8.md) | Client disconnect during chunked prefill stops the prefill engine | bug (crash) | high | reproduced 2/2  none |
| [F-9](F-9.md) | Unaligned prompt lengths give a different first token in PD mode | resolved: not a bug (F-3 artefact) | none | identical 2/2 at c=1  none |
| [F-10](F-10.md) | One 1920×1080 image stops the Qwen3-VL prefill engine | bug (crash) | high | reproduced 2/2  #488 (same mechanism, other trigger); #1068 |
| [F-11](F-11.md) | Chunked prefill deadlocks when the SWA continuation budget hits 0 before eviction (gpt-oss, 1 slot, TPU rounding) | bug (hang) | high | reproduced 4/4, root cause in code  none (#1067 would hit it) |
| [F-12](F-12.md) | P/D handshake never completes; request held on both engines with no deadline | bug (hang) | high | reproduced on 3/3 fresh pairs (one recipe)  #1113/#1078 related (those expire) |
| [F-13](F-13_github_issue.md) | `--kv-canary` kills every TPU engine at start-up (fused pool not supported by the canary) | bug (start-up crash, observability) | medium | 2/2 models  none |

Overlap check: all 69 open issues read on 2026-10-08 (`results/bug_reports/OVERLAP.md`). Production triage (severity, priority, filing order):
`PRIORITIES.md`. Ready-to-file GitHub issue drafts with code traces: `F-1_github_issue.md`, `F-2_github_issue.md`, `F-5_github_issue.md`,
`F-6_github_issue.md`, `F-8_github_issue.md`, `F-10_github_issue.md`, `F-11_github_issue.md`, `F-12_github_issue.md`, `F-13_github_issue.md`.

## Common steps

1. Make sure that no other user holds the TPU: `bash scripts/tpu_free.sh`.
2. Start a server pair: `bash scripts/pd.sh start recipes/<model>_A_128k.json <run-name>`. Wait 4 to 8 minutes.
3. Start a single (monolithic) server: `bash scripts/pd.sh golden recipes/<model>_G_128k_fp8_e4m3.json <run-name> a|b`.
4. Stop all servers that you own: `bash scripts/pd.sh stop`.
5. Logs are in `results/runs/<run-name>/<role>/server.log` on the host that runs the role. Compressed copies are in `results/logs/`.
"""

for fid, text in REPORTS.items():
    open(f"{OUT}/{fid}.md", "w").write(text.lstrip("\n"))
open(f"{OUT}/INDEX.md", "w").write(INDEX.lstrip("\n"))
print(f"wrote {len(REPORTS)} reports + INDEX to {OUT}")
