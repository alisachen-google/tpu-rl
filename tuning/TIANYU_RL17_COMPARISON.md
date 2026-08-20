# tianyu-rl-17 vs g4r1n (native gemma4 4x4x4) — manifest comparison (2026-08-20)

Shared skeleton (our chart descends from their manifest). Deltas that matter:

## Covered in our image (no port needed)
- patch_rpa_blocks: RPA env forwarding — our image has PR #3395; g4p2 probe proved the tuple
  reaches the kernel. THEIR discipline to copy on our RPA rung: verify kernel name in the
  decode profile (RPAd-...-bkv_<sz>_<csz>), else timing void.
- patch_cd_masking v2 + patch_cd_report: CD-window EOS-reporting correctness — upstreamed in
  head-612fcd2e (#3401/d449d04); behaviorally confirmed (EOS-terminating completions, rising rewards).
- patch_reward_parallel: upstreamed into image tunix reward_manager (source of our
  PicklingError→parent-fallback warnings; harmless).

## Adoption candidates
1. Colocated mesh: trainer/sampler_devices_fraction=1.0, rollout dp16×tp8, chips_per_vm=8 —
   doubles devices per phase; rung after control.
2. patch_log_stats (KV-usage gauge via disable_log_stats=False in vllm_sampler _vllm_config) — cheap.
3. TPU_SKIP_MDS_QUERY=true on workers (their START_SESSION fix; zero risk).
4. patch_perf_v2 (Perfetto timeline) — nice-to-have.
5. Fixed gcs_scratch/compile-cache paths for warm restarts — only if stale-RM (Fix #6) is confirmed
   fixed in Pathways 20260730; they run fixed paths without issue.

## Deliberate deviations we keep
- emptyDir /tmp (Fix #7) vs their hostPath.
- Per-render UUID scratch (Fix #6) — cost: cold Pathways compile per re-render.
- safe-to-evict=false annotations (their manifest lacks them; they absorb kills via maxRestarts 5).
- Shifted pathways ports per run (Fix #12/#22) vs their fixed 29000/29001.
- Single profiler (phased only) — profiler=xplane + PHASED together crashed g4dlv2
  ("start_trace called while a trace is already being taken"); they run both, presumably tolerated
  by their patch_vllm_sampler variant + newer pathwaysutils behavior. Re-test before adopting.

## Workload non-comparables
qwen3-0.6b vs gemma4-e2b; 16384/8192 vs 8192/4096; tmbs 32 vs 1 (262k vocab wall; PR#4409 rung
unlocks 2); scan True vs False; their levers RPA=1,1024,1,1024 + EOS_CHECK_INTERVAL=8 + CD depth 128
vs our control (no tuple, interval 0, depth 64). Their validity model (distributional gates: LENDIST
band, osl_mean, reward band) matches ours.
