# GitHub issue draft for sgl-project/sglang-torchtpu

**Title:** [TPU] `--kv-canary` cannot start any engine on TPU: the canary pool patcher has no attacher for `TpuFusedMHATokenToKVPool` (NotImplementedError) and the SWA adapter assumes separate `k_buffer`/`v_buffer` (AttributeError); both crash all ranks at start-up

**Labels:** bug, tpu, observability

---

## Summary

The KV canary (`--kv-canary log|raise`) is the build's own detector for use-after-free and overlapping page writes in the KV pool. On TPU it
cannot be enabled: every scheduler rank dies during memory-pool set-up, 6 s after `Load weight end`, the launcher kills the process group,
and the server never comes up. Reproduced on both of our models, with two different failures that have the same root cause:

| Model | Top-level pool class | What happens |
|---|---|---|
| gpt-oss-120b (hybrid sliding window) | `SWAKVPool` → attacher `attach_swa` | the SWA adapter reads `sub_pool.k_buffer[0]` on each sub-pool; the TPU sub-pools are `TpuFusedMHATokenToKVPool`, which keeps one fused 5-D `kv_buffer` and has no `k_buffer`/`v_buffer` → `AttributeError: 'TpuFusedMHATokenToKVPool' object has no attribute 'k_buffer'. Did you mean: 'kv_buffer'?` |
| Qwen3-VL-30B-A3B (full attention) | `TpuFusedMHATokenToKVPool` | the attacher registry is keyed by exact type and knows only the generic pool classes → `NotImplementedError: kv-canary: no attacher registered for pool class TpuFusedMHATokenToKVPool; supported: ['DeepSeekV4TokenToKVPool', 'MHATokenToKVPool', 'MHATokenToKVPoolFP4', 'SWAKVPool']` |

The failure happens on both the prefill and the decode engine of a disaggregated pair, and would happen on a monolithic server the same way.
Nothing in argument validation rejects the flag on `--device tpu`, so the user gets a crash with `exited with -9` instead of a message.

**Expected behavior.** Either the canary works on TPU (an attacher that understands the fused 5-D layout), or `--kv-canary` other than `none`
is rejected at argument validation on TPU with a clear message. A configuration option must not take every rank down after the weights are loaded.

## Environment

| Item | Value |
|---|---|
| sglang-torchtpu | `main` @ 09d1244c2641c3232ec0fcf5dd2d3bd372d5d709 (2026-10-07), sglang 0.5.13.dev7249 |
| TPU stack | torch-tpu 0.1.1+release.2026.9.22.20260927223741, libtpu 0.0.47, tpu-sync-torch 0.0.1.dev20261002213151, torch 2.11.0+cpu, transformers 5.8.1 |
| Hardware | Two TPU v7x hosts (4 chips / 8 TensorCores each), prefill on host a, decode on host b, TP 8, EP 8 |
| Flags that trigger it | `--kv-canary log --kv-canary-real-data partial --kv-canary-sweep-interval 50` on both engines (`--disaggregation-enable-kv-checksum` was also on; it is unrelated, see below) |
| Models | `openai/gpt-oss-120b` (MXFP4, fp8 KV, page 64); `Qwen/Qwen3-VL-30B-A3B-Instruct` (bf16, fp8 KV, page 128) |

## Code trace

All links at commit 09d1244c26.

**1. The attacher registry knows only the generic pools and is keyed by exact type:**
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/kv_canary/pool_patcher/api.py#L26-L35
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/kv_canary/pool_patcher/api.py#L48-L55

`register_pool_attacher` exists but no TPU module calls it (no match for it outside `api.py`), so the TPU pool subclass is never registered.

**2. The TPU pool is a subclass with a fused layout.** `TpuFusedMHATokenToKVPool(MHATokenToKVPool)` builds one 5-D `kv_buffer` per layer for the
Pallas kernel instead of `k_buffer`/`v_buffer`:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/hardware_backend/tpu/kv_pool.py#L71-L90

**3. The SWA adapter assumes the generic layout.** For `SWAKVPool` (gpt-oss) the adapter is found, but it reads `k_buffer[0]` and `v_buffer[0]`
of each sub-pool to size the canary buffers and to build the real-data row sources:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/kv_canary/pool_patcher/adapters/swa.py#L11-L80

**4. Tracebacks (identical on all 8 ranks of both engines):**

gpt-oss:
```
  File ".../sglang/srt/kv_canary/pool_patcher/api.py", line 58, in attach_canary_buffers
    groups = attacher(
  File ".../sglang/srt/kv_canary/pool_patcher/adapters/swa.py", line 22, in attach_swa
    full_group = _build_subpool_group(
  File ".../sglang/srt/kv_canary/pool_patcher/adapters/swa.py", line 65, in _build_subpool_group
    num_slots = int(sub_pool.k_buffer[0].shape[0])
AttributeError: 'TpuFusedMHATokenToKVPool' object has no attribute 'k_buffer'. Did you mean: 'kv_buffer'?
[2026-10-08 17:30:28] Received sigquit from a child process. It usually means the child failed.
RuntimeError: An engine or router exited with -9
```

Qwen3-VL:
```
NotImplementedError: kv-canary: no attacher registered for pool class TpuFusedMHATokenToKVPool;
supported: ['DeepSeekV4TokenToKVPool', 'MHATokenToKVPool', 'MHATokenToKVPoolFP4', 'SWAKVPool']
```

## How we detected it

Test case TC-15 of our validation campaign: run Config A for each model with the build's own KV-integrity instrumentation enabled on both
engines (`--disaggregation-enable-kv-checksum` plus the canary in `log` mode), then TC-1 at 8k/32k/128k, concurrency 4 and decode-phase
aborts, and count `KV checksum mismatch` and canary reports. Both pairs failed to come up; the harness captured the tracebacks above. We are
re-running TC-15 with the checksum only, which does not depend on the canary code path.

## How to reproduce

1. Start any model on TPU with `--kv-canary log` (one monolithic server is enough):
   `python -m sglang.launch_server --device tpu --model-path openai/gpt-oss-120b --tp-size 8 --kv-canary log --kv-canary-real-data partial`
2. Watch the log for about 10 s after `Load weight end`.

**Observed.** All ranks raise during `attach_canary_buffers`; the launcher reports `An engine or router exited with -9`.
**Expected.** The server starts with the canary active, or exits at argument parsing with "kv-canary is not supported on TPU".

## Fix proposal

1. **Reject up front.** In server-argument validation, when `device == "tpu"` and `kv_canary != "none"`, raise a `ValueError` that names the
   flag and the reason, until an attacher exists. This is a two-line change and removes the start-up crash.
2. **Support the fused layout.** Register `TpuFusedMHATokenToKVPool` with an attacher (via `register_pool_attacher` from the TPU hardware
   backend) that sizes the canary buffers from `kv_buffer[0].shape[0]` and builds the real-data row sources from the K and V halves of the fused
   row (the pool already exposes `get_fused_kv_buffer`). Make `_build_subpool_group` in the SWA adapter use the same accessor, so hybrid-SWA
   models on TPU work too.
3. **Fail softer.** An exception from `attach_canary_buffers` during pool set-up should disable the canary with an error log rather than kill the
   scheduler, since the canary is diagnostic instrumentation, not part of serving.

## Related

- None of the open issues mentions `kv-canary`. #1068 (fail tests on serving-time recompilation) is the nearest in spirit: both are about
  making the build's own correctness instrumentation usable in TPU CI.

## Evidence

In `alisachen-google/tpu-rl`, directory `sglang_inference/`: `results/runs/A_gpt-oss_128k_kvcheck/decode/server.log` and
`results/runs/A_qwen3-vl_128k_kvcheck/decode/server.log` (tracebacks, all ranks; prefill logs on host a identical), `results/phase_kv_integrity.log`,
`recipes/*_A_128k_kvcheck.json`, `results/FINDINGS.md` (F-13).
