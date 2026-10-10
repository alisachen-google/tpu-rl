## Problem

One ordinary `/generate` request with `return_logprob: true` and `logprob_start_len: 0` kills every TPU engine with the default settings (#1244; reproduced 2/2 on gpt-oss-120b, monolithic and P/D). The compiled forward shim decides per batch whether input (prompt) logprobs can be served: off by default behind `SGLANG_TPU_ENABLE_PROMPT_LOGPROB`, and never under DP attention or pipeline parallelism. It raised `NotImplementedError` from inside the model forward, nothing catches an exception there, and the scheduler process ended on every rank. The client saw a 500 from a dead server; the same request is a clean 400 on other backends.

## Change (3 files, TPU logic in the TPU subdir)

- **New** `hardware_backend/tpu/prompt_logprob.py`: `prompt_logprob_rejection()` is the one answer to "can this server serve prompt logprobs" (flag, DP attention, pipeline parallelism), moved verbatim out of the shim; `prompt_logprob_request_rejection(return_logprob, logprob_start_len, input_len)` applies it per request, mirroring the batch-level test the shim uses (a prompt position is requested when `logprob_start_len` lies before the end of the prompt; `None`, `-1` and the prompt length mean output logprobs only).
- `compilation/compiled_forward_shim.py`: `_prompt_logprob_rejection` delegates to the shared answer. The raise inside the forward stays as a last line of defense for a request that bypassed intake.
- `managers/tokenizer_manager.py` (shared path, 8 lines guarded by `current_platform.is_tpu()`): `_validate_one_request` calls the request-level check next to the existing `logprob_start_len` validation and raises `ValueError`, which the HTTP layer turns into **400** with the same message the shim produced.

## Tests

Added to the existing files. `test/registered/unit/hardware_backend/tpu/test_compiled_forward_shim.py`: the shared answer for flag off, flag on, DP attention and pipeline parallelism; the request-level predicate for `None`, `-1`, the prompt length, beyond it, and a prompt position; the shim delegates to the shared answer. `test/registered/unit/managers/test_sampling_mask_validation.py`: the tokenizer manager refuses prompt logprobs with the flag off (start 0 and 2), admits output-only logprobs (`-1`, prompt length, `return_logprob=False`), admits prompt logprobs with the flag on, and leaves other platforms alone.

```
pytest test/registered/unit/hardware_backend/tpu/test_compiled_forward_shim.py test/registered/unit/managers/test_sampling_mask_validation.py
131 passed, 66 subtests passed
```

## Live validation

Running on a TPU v7x host (monolithic gpt-oss-120b, `--enable-torch-compile`): this branch with the flag off (expect 400 and a healthy engine), this branch with the flag on (expect prompt logprobs served), and upstream main with the flag off (the crash). Results will be added here before the PR leaves draft.

Fixes #1244
