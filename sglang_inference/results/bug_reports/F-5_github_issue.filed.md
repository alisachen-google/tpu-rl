Filed as #1244 https://github.com/sgl-project/sglang-torchtpu/issues/1244

## Summary

The TPU compiled backend does not serve prompt (input) logprobs unless `SGLANG_TPU_ENABLE_PROMPT_LOGPROB=1` is set. That limitation is known
(#432 asks for the feature). What #432 does not say is what happens when a client asks anyway: the backend raises `NotImplementedError` from
inside the compiled forward shim, during `run_batch`, on every tensor-parallel rank. The scheduler loop does not catch it, all eight ranks exit,
the launcher kills the process group, and in a disaggregated deployment the load balancer goes down with the prefill engine. Reproduced 2 of 2.

A single ordinary `/generate` call with `"return_logprob": true, "logprob_start_len": 0` is enough. There is no authentication or size threshold
involved, so any client can take the engine down.

**Expected behavior.** The request is rejected at validation time with HTTP 400 and the same message the shim produces today ("input/prompt
logprobs (echo) need SGLANG_TPU_ENABLE_PROMPT_LOGPROB=1 ..."). The engine serves the next request. Failures that are a property of one request
never leave the scheduler loop.

## Environment

| Item | Value |
|---|---|
| sglang-torchtpu | `main` @ 6187d8bbc9b32d8f49be8be962dc30452169910a (2026-10-07), sglang 0.5.13.dev7249 |
| TPU stack | torch-tpu 0.1.1+release.2026.9.22.20260927223741, libtpu 0.0.47, tpu-sync-torch 0.0.1.dev20261002213151, torch 2.11.0+cpu, transformers 5.8.1 |
| Hardware | Two TPU v7x hosts (4 chips / 8 TensorCores each), prefill on host a, decode on host b, TP 8, EP 8. A monolithic server is affected the same way |
| Model | `openai/gpt-oss-120b` (MXFP4), KV cache fp8_e4m3, `--enable-torch-compile`, `SGLANG_TPU_ENABLE_PROMPT_LOGPROB` unset (default) |

## Code trace

All links at commit 6187d8bbc9 (current main, 2026-10-09); first observed at 09d1244c26.

**1. The feature flag.** Prompt logprobs on the compiled TPU path are gated by an env var, default off:
https://github.com/sgl-project/sglang-torchtpu/blob/6187d8bbc9b32d8f49be8be962dc30452169910a/python/sglang/srt/environ.py#L1907

**2. Request validation lets the request through.** The tokenizer manager already validates `logprob_start_len` for one special case
(encoder SWA replay) and raises `ValueError`, which becomes a clean HTTP 400. There is no equivalent check for the TPU prompt-logprob gate:
https://github.com/sgl-project/sglang-torchtpu/blob/6187d8bbc9b32d8f49be8be962dc30452169910a/python/sglang/srt/managers/tokenizer_manager.py#L1286-L1292

**3. The rejection is computed inside the forward.** The compiled forward shim decides per batch whether input logprobs can be served and
raises if not:
https://github.com/sgl-project/sglang-torchtpu/blob/6187d8bbc9b32d8f49be8be962dc30452169910a/python/sglang/srt/hardware_backend/tpu/compilation/compiled_forward_shim.py#L772-L810
https://github.com/sgl-project/sglang-torchtpu/blob/6187d8bbc9b32d8f49be8be962dc30452169910a/python/sglang/srt/hardware_backend/tpu/compilation/compiled_forward_shim.py#L964-L967

The docstring says the check is "consulted only for requests that actually ask for them, so none of this lands on anyone else's forward". That
is true for the cost, but not for the consequence: the exception propagates through `run_batch` and ends the scheduler process, which ends
every other request too.

**4. Traceback (identical on all ranks):**

```
[TPx] Scheduler hit an exception: Traceback (most recent call last):
  File ".../sglang/srt/disaggregation/prefill.py", line 773, in event_loop_overlap_disagg_prefill
  File ".../sglang/srt/managers/scheduler.py", line 4670, in run_batch
  File ".../sglang/srt/managers/tp_worker.py", line 646, in forward_batch_generation
  File ".../sglang/srt/model_executor/model_runner.py", line 1783, in forward
  ...
  File ".../sglang/srt/hardware_backend/tpu/compilation/compiled_forward_shim.py", line 967, in _forward
    raise NotImplementedError(rejection)
NotImplementedError: input/prompt logprobs (echo) need SGLANG_TPU_ENABLE_PROMPT_LOGPROB=1 on the TPU compiled backend;
only output (next-token) logprobs are served by default.
```

followed by `Rank 0 scheduler died` and the launcher's `exited with -9`.

## How we detected it

We run a test campaign on a 1-prefill + 1-decode pair with gpt-oss-120b. One part of it sends each sampling option once and then checks that both engines still answer `/health`.

The `prompt_logprobs` case sends this request:

```json
{"text": "The three primary colors are", "return_logprob": true, "logprob_start_len": 0,
 "sampling_params": {"temperature": 0, "max_new_tokens": 32}}
```

What we saw:

- The request came back as HTTP 500 after 65 s. That is the load balancer giving up on a prefill engine that no longer answers.
- `/health` on the prefill engine returned nothing; the port refused connections. The decode engine was still fine.
- The prefill log showed the traceback above, once per rank.
- We repeated it on a fresh pair the next day with the same result.
- The same request without `logprob_start_len` (output logprobs only) works. The problem is specific to prompt logprobs.

## How to reproduce

1. Start gpt-oss-120b on TPU with `--enable-torch-compile` and the default environment (a monolithic server is enough).
2. Send:
   ```
   curl -s $SERVER/generate -H 'Content-Type: application/json' \
     -d '{"text":"The three primary colors are","return_logprob":true,"logprob_start_len":0,"sampling_params":{"temperature":0,"max_new_tokens":32}}'
   ```
3. `curl -s -o /dev/null -w '%{http_code}\n' $SERVER/health`.

**Observed.** Step 2 returns HTTP 500 (or hangs until the proxy times out). Step 3 returns `000`. The log shows the `NotImplementedError` on every rank.
**Expected.** Step 2 returns HTTP 400 with the message; step 3 returns `200`.

## Fix proposal (PR to follow)

1. **Reject at validation.** In `TokenizerManager._validate_one_request` (next to the existing `logprob_start_len` check), when the model runner
   is the TPU compiled backend and `SGLANG_TPU_ENABLE_PROMPT_LOGPROB` is off, raise `ValueError` for `return_logprob` with
   `logprob_start_len` not in `(None, -1, len(input_ids))`. Reuse the message string from `_prompt_logprob_rejection` so the two stay in sync.
   The same applies to the DP-attention and pipeline-stage rejections in that function: all three are static properties of the server and can
   be decided before the request is scheduled.
2. **Never let a per-request error end the scheduler.** As a second line of defense, convert `NotImplementedError` from the forward shim into a
   per-request failure (abort the requests of the batch with HTTP 500 and the message) instead of letting it reach the event loop. This covers
   future gates added to the shim.
3. Optionally resolve #432 by enabling the feature; until then the server should refuse the request cleanly rather than crash.

## Related

- #432 (feature request, Aug 19): prompt logprobs not supported on TPU. The feature has since landed behind the flag (PR #597, Sep 1), so #432 is
  about enabling it; this report is about the crash when a client asks while the flag is off (the default), and applies equally to the DP-attention
  and pipeline-parallel rejections in the same function when the flag is on.
- PR #638 (Sep 4) fixed the sibling crash on the enabled path (prompt logprobs + top-k). PR #137 (Jul 14) introduced the narrowed guard that this
  report trips.

## Evidence

In `alisachen-google/tpu-rl`, directory `sglang_inference/`: `results/A/params/gpt-oss/params_pd_A.json` (case `prompt_logprobs`),
`results/phase_A_gptoss_redo.log` (section "(5) F-5 repro"), prefill logs under `results/logs/runs/A_gptoss_128k_run*`, `results/bug_reports/F-5.md`.