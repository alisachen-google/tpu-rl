## Live validation

gpt-oss-120b, single engine (fp8 e4m3 recipe), head 79b242fa38 (a2c09c3b1d for the flag-off run; same code, docstring-only difference). Each case: fresh server, four probes, then a health check, one follow-up request and a burst of 8 plain requests.

| Case | Probe | Result |
|---|---|---|
| main, flag off (measured earlier) | `return_logprob=true, logprob_start_len=0` | connection dropped (HTTP 000); `NotImplementedError` in all 8 ranks; `/health` dead afterwards; follow-up request HTTP 000 |
| fix, flag off | plain request | 200 |
| fix, flag off | output-only logprobs (`return_logprob=true`) | 200, 8 output logprobs, 0 input logprobs |
| fix, flag off | `logprob_start_len=0` | **400** `input/prompt logprobs (echo) need SGLANG_TPU_ENABLE_PROMPT_LOGPROB=1 ...` |
| fix, flag off | `logprob_start_len=2` | **400**, same message |
| fix, flag off | after: health / follow-up / burst of 8 | 200 / 200 / 8×200; scheduler logged the two refusals as errors, no exception |
| fix, flag on (`SGLANG_TPU_ENABLE_PROMPT_LOGPROB=1`) | plain, output-only logprobs | 200, 200 (8 output logprobs) |
| fix, flag on | `logprob_start_len=0` and `=2` | 200, 200; 5 input logprobs returned for the 5-token prompt |
| fix, flag on | after: health / follow-up / burst of 8 | 200 / 200 / 8×200 |

The 4 tracebacks in each server log are the launcher's post-warmup `freeze_gc` request hitting a port that is not yet open; they are present in every run of this recipe and unrelated to this change.
