# GitHub issue draft for sgl-project/sglang-torchtpu

**Title:** [TPU] Server aborts with SIGABRT (`exit code: -6`) when the shared tier-2 compile cache in `/dev/shm/torch_tpu_cache` is owned by another user; the launcher swallows the error and the default cache name is shared by everyone

**Labels:** bug, tpu, robustness

---

## Summary

torch_tpu keeps a tier-2 compile cache under `/dev/shm/torch_tpu_cache/<name>/`. On a shared host the directory is created by whichever
user starts first, with mode 755. Every later start by a different user then fails: the launcher tries to create the per-run cache directory,
ignores the `OSError`, and starts the engine; torch_tpu hits the same permission error and treats it as fatal (`CHECK failed`), killing every
rank with `SIGABRT`. SGLang reports only `Rank 0 scheduler died during initialization (exit code: -6)`; the real cause is eight lines deeper in the
TPU runtime log. Reproduced 2 of 2, in two forms:

1. With a per-run tier-2 name: `Check failed: status.ok() Failed to create tier-2 cache directory: /dev/shm/torch_tpu_cache/<name>/<hash>`.
2. With no tier-2 name (`ENABLE_COMPILE_CACHE=0` in our harness, or any launch that leaves `TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE` unset):
   torch_tpu falls back to the name `default`, so all users share `/dev/shm/torch_tpu_cache/default/<hash>/`, and the second user fails on the
   lock file: `Check failed: lock_fd_.valid() Failed to create tier-2 cache entry lock file: ... Permission denied`.

So "cache off" is not a work-around, and two users with default settings cannot share a host. The only way out is `chmod 1777` on the root
directory plus a unique tier-2 name per user or job.

**Expected behavior.** The server starts without the tier-2 cache (slower compile, correct results) when the directory is not writable, or it
stops before loading weights with a one-line error that names the directory and the owner. The launcher must not pass a cache name whose
directory it failed to create.

## Environment

| Item | Value |
|---|---|
| sglang-torchtpu | `main` @ 09d1244c2641c3232ec0fcf5dd2d3bd372d5d709 (2026-10-07), sglang 0.5.13.dev7249 |
| TPU stack | torch-tpu 0.1.1+release.2026.9.22.20260927223741, libtpu 0.0.47, tpu-sync-torch 0.0.1.dev20261002213151 |
| Hardware | TPU v7x host shared by two users, `/dev/shm` tmpfs |
| Models | any; seen with `Qwen/Qwen3-VL-30B-A3B-Instruct` and `openai/gpt-oss-120b`, `--enable-torch-compile` |

## Code trace

All links at commit 09d1244c26.

**1. The launcher creates the directory and ignores failure.** `prepare_runtime_env` wraps the `mkdir` in `try/except OSError: pass` and keeps
the env var set:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/benchmark/tpu/launch_server.py#L659-L684

The in-tree env override does the same, with the comment "The tier-2 cache is optional, as in the shell launcher":
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/python/sglang/srt/hardware_backend/tpu/env_override.py#L18-L25

**2. torch_tpu treats the same failure as fatal.** The runtime (outside this repository, `tier2_compilation_cache.cc`) `CHECK`s the directory
and lock-file creation; a failure raises `SIGABRT` in every scheduler process. The log shows, once per rank:

```
F0000 ... tier2_compilation_cache.cc:241] Check failed: status.ok() Failed to create tier-2 cache directory:
/dev/shm/torch_tpu_cache/pd_golden_qwen3-vl_G_128k_fp8_e4m3/ee8e2de63f954d59 with error: ...
```
or
```
F0000 ... tier2_compilation_cache.cc:266] Check failed: lock_fd_.valid() Failed to create tier-2 cache entry lock file:
/dev/shm/torch_tpu_cache/default/ee8e2de63f954d59/<entry>.lock with error: Permission denied
```

**3. SGLang reports the symptom, not the cause.**

```
RuntimeError: Rank 0 scheduler died during initialization (exit code: -6)
```

**4. The default name is shared.** The launcher sets the tier-2 name from `TORCH_TPU_TIER2_NAME` or `torchtpu_<model_slug>`; when neither
applies, torch_tpu's own default is the literal `default`, with no per-user component:
https://github.com/sgl-project/sglang-torchtpu/blob/09d1244c2641c3232ec0fcf5dd2d3bd372d5d709/benchmark/tpu/launch_server.py#L176-L182

## How we detected it

Our two TPU hosts are shared with a colleague's campaign. The first server start of our campaign on host b failed within 10 seconds of
`Load weight begin` with `exit code: -6`. The TPU runtime lines above were found by searching the log for `Check failed`.
`ls -ld /dev/shm/torch_tpu_cache` showed the directory owned by the other user with mode 755. After `chmod 1777` and a per-run tier-2 name
every start succeeded. Setting `ENABLE_COMPILE_CACHE=0` in our launcher the next day reproduced the second form, which is how we learned that
the unset case shares a single `default` directory.

## How to reproduce

1. As user A: `mkdir -p /dev/shm/torch_tpu_cache && chmod 755 /dev/shm/torch_tpu_cache`.
2. As user B: start any model with `--enable-torch-compile` through `benchmark.tpu.launch_server` (or directly), with
   `TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=myrun` for form 1 or unset for form 2.
3. Watch the log for about 10 s after `Load weight begin`.

**Observed.** All ranks die with `SIGABRT`; the top-level error is `Rank 0 scheduler died during initialization (exit code: -6)`; the cause is in
the `F0000 ... tier2_compilation_cache.cc` lines.
**Expected.** Either a start without the tier-2 cache and an INFO line saying so, or a clear error before weights load.

## Fix proposal

1. **Launcher:** when creating the tier-2 directory fails, either unset `TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE` (and log that the cache is
   disabled) or exit with the `OSError` text. Do not keep a name whose directory is not writable. Apply the same to `env_override.py`.
2. **Per-user default name:** include the uid or `$USER` in the default tier-2 name (`default-<uid>`), so independent users never share a
   directory, and create `/dev/shm/torch_tpu_cache` with mode `1777` when the launcher creates it.
3. **torch_tpu (separate repository):** fall back to no tier-2 cache with a warning instead of `CHECK`-failing when the directory or lock file
   cannot be created. Report that upstream; this issue tracks the launcher side.
4. **Diagnostics:** when a scheduler dies during initialization with a signal, print the last `F0000` line from the TPU log in the
   `RuntimeError` message so the cause is visible without searching.

## Related

- None of the 69 open issues mentions `/dev/shm/torch_tpu_cache` or the tier-2 `CHECK`.

## Evidence

In `alisachen-google/tpu-rl`, directory `sglang_inference/`: `results/FINDINGS.md` (F-1 entries, both forms), `results/bug_reports/F-1.md`,
`scripts/launch_role.sh` (the per-role tier-2 name and the `chmod 1777` work-around we apply).
