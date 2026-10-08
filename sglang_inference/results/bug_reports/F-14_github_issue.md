# GitHub issue draft for sgl-project/sglang-torchtpu

**Title:** [TPU/PD] `--disaggregation-enable-kv-checksum` is CUDA-only: on TPU the prefill engine passes health checks and then every rank dies on the first real request (`assert kv_page_indices_gpu.is_cuda`)

**Labels:** bug, tpu, pd

---

## Summary

The P/D transfer checksum (`--disaggregation-enable-kv-checksum`, Adler-32 over each request's KV pages on the prefill, verified on the
decode) is documented as "Enable on both prefill and decode engines" with no platform restriction. On TPU it cannot work and it fails in the
worst possible way: the prefill engine starts, loads weights, compiles, answers `/health` and `/health_generate` (health-check requests skip
the checksum and send `0`), and then the first real request kills all eight scheduler ranks with

```
  File ".../sglang/srt/disaggregation/prefill.py", line 1418, in send_kv_chunk
  File ".../sglang/srt/disaggregation/checksum.py", line 105, in compute
    assert kv_page_indices_gpu.is_cuda and kv_page_indices_gpu.is_contiguous()
AssertionError
```

Reproduced on gpt-oss-120b and on Qwen3-VL-30B-A3B (identical traceback on all ranks of both). The checksum computer is constructed with `torch.device("cuda:…")`,
asserts `is_cuda` on its inputs and calls the CUDA Adler-32 kernels in `sglang.kernels.ops.memory.adler32`; there is no TPU implementation and
no validation that rejects the flag on `--device tpu`.

**Expected behavior.** Either a TPU implementation of the strided Adler-32 (or a host-side fallback), or an argument-validation error on TPU.
A monitoring option must never let an engine pass its health checks and then die on the first user request.

## Environment

| Item | Value |
|---|---|
| sglang-torchtpu | `main` @ 09d1244c2641c3232ec0fcf5dd2d3bd372d5d709 (2026-10-07), sglang 0.5.13.dev7249 |
| TPU stack | torch-tpu 0.1.1+release.2026.9.22.20260927223741, libtpu 0.0.47, tpu-sync-torch 0.0.1.dev20261002213151, torch 2.11.0+cpu, transformers 5.8.1 |
| Hardware | Two TPU v7x hosts (4 chips / 8 TensorCores each), prefill on host a, decode on host b, TP 8, EP 8, raiden transfer |
| Flags that trigger it | `--disaggregation-enable-kv-checksum` on both engines (Config A recipe otherwise; `--kv-canary none`) |
| Models | `openai/gpt-oss-120b` (MXFP4, fp8 KV, page 64); `Qwen/Qwen3-VL-30B-A3B-Instruct` (bf16, fp8 KV, page 128) |

## Code trace

All links at commit 09d1244c26.

**1. Documented without a platform restriction:**
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/arg_groups/fields/disagg.py#L66-L72

**2. The prefill constructs the computer on a CUDA device unconditionally:**
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/disaggregation/prefill.py#L230-L240

**3. Health-check requests skip the computation, every other request runs it on the last chunk:**
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/disaggregation/prefill.py#L1405-L1420

**4. `compute` asserts a CUDA tensor and calls the CUDA kernel:**
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/disaggregation/checksum.py#L98-L115
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/kernels/ops/memory/__init__.py#L56-L66

Nothing under `hardware_backend/tpu/`, `test/registered/tpu/` or `benchmark/tpu/` references the checksum, so it has never run on TPU.

**5. Consequence.** The `AssertionError` escapes `process_batch_result_disagg_prefill`; all ranks exit; the launcher kills the group and the
load balancer with it. In our run the engine had been healthy for two minutes and had answered the launcher's warm-up requests.

## How we detected it

Test case TC-15 of our validation campaign (P/D KV-transfer integrity with the build's own instrumentation). The pair started normally
(`prefill /health ready after 136 s`, `LB /health_generate ready`), the harness sent the first 8k TC-1 prompt, and every request from then on
returned connection errors. The prefill log shows two warm-up `Prefill batch` lines (`#new-token: 1`) followed by `Scheduler hit an exception`
on all eight ranks at 18:24:52 with the traceback above. The decode engine stayed healthy.

## How to reproduce

1. Start a gpt-oss-120b P/D pair on TPU with `--disaggregation-enable-kv-checksum` on both engines (any other flags as in a working recipe).
2. Wait for `LB /health_generate ready`.
3. Send one normal request: `curl -s $LB/generate -d '{"text":"hello","sampling_params":{"max_new_tokens":8}}' -H 'Content-Type: application/json'`.
4. `curl -s -o /dev/null -w '%{http_code}\n' $PREFILL/health`.

**Observed.** Step 3 returns an error after the proxy timeout; step 4 returns `000`; the prefill log shows the assertion on every rank.
**Expected.** Step 3 answers; or the server refuses to start with "kv checksum is not supported on TPU".

## Fix proposal

1. **Reject up front.** In argument validation, `disaggregation_enable_kv_checksum` with `device == "tpu"` → `ValueError` naming the flag,
   until a TPU implementation exists. Also skip the `torch.device("cuda:…")` construction on non-CUDA devices.
2. **Implement for TPU.** The checksum only needs the page bytes of one request; a host-side Adler-32 over the pages gathered from the TPU
   pool (or a Pallas kernel) would make the transfer check usable where it matters most, on a new transport backend (raiden). Until then, the
   only transfer-integrity evidence on TPU is the differential test against the monolithic server.
3. **Never die on a per-request failure.** As with our other reports (F-5, F-8, F-10), an exception while post-processing one request should
   fail that request, not the scheduler.

## Related

- No open issue mentions the KV checksum. #1066 and #1067 (P0 long-context P/D e2e tests) would benefit from the checksum being usable on TPU.
- Our F-13 (`--kv-canary` also unusable on TPU) is the companion report: the two built-in KV integrity checks both fail on TPU, one at
  start-up and one on the first request.

## Evidence

In `alisachen-google/tpu-rl`, directory `sglang_inference/`: prefill logs of `A_gpt-oss_128k_kvcheck_nocanary` and `A_qwen3-vl_128k_kvcheck_nocanary`
(compressed copies in `results/logs/runs/`), `results/phase_kv_checksum_gptoss.log`, `results/phase_kv_checksum_qwen.log`,
`recipes/*_A_128k_kvcheck_nocanary.json`, `results/FINDINGS.md` (F-14).
