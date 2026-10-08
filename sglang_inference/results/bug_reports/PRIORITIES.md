# Issue triage for production: priority and severity

Scope: findings F-1 to F-12 from the 1-prefill + 1-decode validation campaign on TPU v7x (sglang-torchtpu `main` @ 09d1244c26, 2026-10-07/08).
Overlap checked against the 69 open issues of `sgl-project/sglang-torchtpu` captured 2026-10-08 01:30 UTC (`OVERLAP.md`).

## Scales

**Severity (impact on a production deployment if hit)**

| Severity | Meaning |
|---|---|
| S0 | Engine outage or data corruption triggered by ordinary client traffic; affects all tenants |
| S1 | A request or slot is lost or held indefinitely with no error; capacity leaks until restart |
| S2 | Wrong or degraded service for a feature or configuration, with a work-around |
| S3 | Operational friction; no effect on correctness or availability once configured |

**Priority (order to fix, for the repo's P0/P1/P2 title prefixes)**

| Priority | Meaning |
|---|---|
| P0 | Blocks production use; any client can trigger it; fix before the next deployment |
| P1 | Must fix for the targeted configurations (long context, multi-tenant, multimodal); work-around exists or trigger is narrower |
| P2 | Fix when touching the area; documented work-around |
| P3 | Nice to have; environment or diagnostics |

## Triage table

| ID | Title | Repro | Severity | Priority | Action | Why this rating |
|---|---|---|---|---|---|---|
| F-8 | Client disconnect during chunked prefill kills the prefill engine (KV row released twice) | 2/2 | **S0** | **P0** | File `F-8_github_issue.md` | A disconnect is the most common client event there is. One such event ends every in-flight request on the engine. Root cause located; fix is small |
| F-5 | One prompt-logprob request kills the engine (rejection raised inside the forward) | 2/2 | **S0** | **P0** | File `F-5_github_issue.md`, reference #432 | Any client with a plain JSON body takes the engine down. Fix is a validation check that already exists for a sibling case |
| F-10 | One large image kills the Qwen3-VL prefill engine (recompile under `fail_on_recompile`) | 2/2 | **S0** | **P0** for multimodal deployments | File `F-10_github_issue.md`, reference #488, #1068 | Default config accepts images the compiled range cannot serve. Sibling vision policies already have the eager fallback; porting it is the fix |
| F-12 | P/D handshake never completes; request held on both engines with no deadline | 3/3 fresh pairs on one recipe | **S1** | **P0** for PD deployments | File `F-12_github_issue.md`, reference #1113, #1078 | Deterministic for a recipe we cannot yet explain. The raiden receiver has no deadline while bootstrapping, so any handshake fault is a permanent hold (our recipe's 7200 s timeouts also hide the sender's deadline). The deadline fix is independent of the trigger |
| F-11 | Chunked prefill deadlocks when the SWA continuation budget hits 0 before eviction (gpt-oss, 1 slot, TPU rounding) | 4/4; control with 4 slots passes | **S1** | **P1** | File `F-11_github_issue.md`, reference #1067 | Hits exactly the 1M-context single-slot configuration the long-context work targets. Root cause fully located; three fixes proposed. Narrow trigger (one slot) keeps it below P0 |
| F-6 | `n>1` never returns through the pair; all sub-requests share one bootstrap room | 3/3 | **S1** | **P1** | File `F-6_github_issue.md` | Parallel sampling is a standard API feature; each call leaks a decode slot until the client disconnects. Work-around: clients send n separate requests. Fix is an indexing change in the tokenizer manager |
| F-2 | Qwen3-VL-30B MoE decoder never compiled on TPU (6 tok/s instead of 180) | every start | **S2** | **P1** | File `F-2_github_issue.md` with patch, reference #1069 | 30× decode slowdown with no warning; patch validated for speed (180 tok/s) and accuracy (GSM8K 0.99 vs 0.97 eager), 0 serving-time recompiles. Not an outage, hence S2 |
| F-1 | SIGABRT when the shared tier-2 cache directory is owned by another user; default name shared by all users | 2/2 | **S3** | **P3** | File `F-1_github_issue.md` | Blocks every second user on a shared host, but single-tenant production hosts are unaffected and `chmod 1777` plus a unique cache name works around it. The misleading `exit code -6` costs debugging time |
| F-7 | Decode recovers ~2 min after the prefill engine dies | observed | S2 | — | **Comment on #1113** (same mechanism) | Duplicate in mechanism; our data adds the full-engine-death case |
| F-4 | PD vs monolithic outputs differ for long gpt-oss prompts | 6/6 | S3 (informational) | — | **Closed**: numerical, no accuracy impact (GSM8K 0.98 vs 0.99); identical with `--disable-hybrid-swa-memory` | Chunk schedule depends on SWA pool headroom; documented in the F-11 issue as a side observation |
| F-9 | Unaligned prompt lengths give a different first token in PD mode | 4/4 → 0/2 at c=1 | — | — | **Closed**: not a bug | Artefact of comparing against a concurrency-8 golden (F-3) |
| F-3 | Greedy decoding is not batch-invariant | measured | — | — | **No issue**; test-method note | Known property of the backend (#321, #391 related); comparisons must be done at equal batch composition |

## Filing order

1. F-8, F-5, F-10 (S0, single-request engine kills; all have complete drafts and located root causes).
2. F-12 (deadline fix is clear even though the trigger is not), then F-11 and F-6.
3. F-2 with the patch.
4. F-1; comment on #1113 for F-7.

## Suggested title prefixes for the repo

The repository marks priority in titles (`P0 [TPU/PD/Raiden] ...`). Proposed: `P0` for F-8, F-5, F-10, F-12; `P1` for F-11, F-6, F-2; no prefix for F-1.

## Cross-cutting recommendations (one sentence each, for the maintainers)

- A per-request failure must never escape the scheduler event loop; F-5, F-8 and F-10 are three different exceptions with the same outcome.
- Every wait in the P/D path needs a deadline that is armed from the moment the request exists, not from a later state transition (F-12, F-6).
- The TPU SWA pool alignment should round up, and chunked-prefill eviction should run before the continuation budget is computed (F-11).
