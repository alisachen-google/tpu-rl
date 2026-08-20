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

## Workload-knob audit: did we follow tianyu's config? (2026-08-20)

FOLLOWED (recipe core): rl.num_generations=8, grpo_beta=0.0, grpo_epsilon=0.2, epsilon_high=0.28,
lr=1e-6, temp/top_k/top_p = 0.8/50/0.95, batch_size=256, rollout_micro_batch_size=256,
gradient_clipping_threshold=1.0, async_scheduling=False, math_verify_num_procs=32, max_num_seqs=256,
enable_dp_attention=False, allow_split_physical_axes=True, enable_tunix_perf_metrics=True,
checkpoint_storage_use_ocdbt=False, the 27 proxy XLA flags @ scoped_vmem 65536, maxRestarts=5,
queue multislice-queue, priority medium, and the 5 stack patches.

DEVIATIONS FORCED BY THE MODEL vs BY OUR CURRENT SETUP (corrected 2026-08-20 — the earlier
"gemma4 physics" label was too strong):
- scan_layers True→False: ARCHITECTURAL. gemma4 layers share KV caches across layers; nn.scan needs
  identical layer bodies over a carried state, so it cannot express it. maxtext's PARAM_MAPPING for
  gemma4 assumes scan_layers=False.
- tmbs 32→1: MEASURED LIMIT OF OUR CURRENT SHARDING, not a law. Logits are [tmbs, seq, vocab] f32 =
  8.59 GB/sequence at 8192x262144; t6g9 measured 130.13 G temporaries at tmbs=4 vs 94.74 G available
  (~32 G/sequence), i.e. cost scales linearly with tmbs => the tensor is REPLICATED, not batch-sharded.
  Our trainer mesh is (data=1, fsdp=64), so with tmbs=1 there is no batch axis to shard along.
  tianyu's qwen3 is larger per sequence (16384x151936x4 = 9.96 GB) yet runs tmbs=32, so on their path
  the logits do shard. OPEN LEAD: make the gemma4 logits shard (maxtext num_vocab_tiling for the vocab
  axis; check the activation_batch axis-rule mapping). PR #4409 sequence-chunking already lifted us to
  tmbs=2 (g4n13s1205), which is the same problem attacked from the sequence axis.
- max_target/prefill 16384/8192→8192/4096: MEMORY-DRIVEN CHOICE inherited from the team e4b recipe
  (head_dim 256 doubles KV/token vs qwen3's 128; 262k vocab inflates logits). Not impossible, just
  costlier; revisit together with the sharding lead.
- max_num_batched_tokens 16640→8448: NOT a deviation — same formula (max_target_length + kv_cache_buffer).

CHOSEN DEVIATIONS — open rungs:
| knob | tianyu | ours | note |
|---|---|---|---|
| trainer/sampler_devices_fraction | 1.0 / 1.0 (colocated, 128 core-devices per phase) | 0.5 / 0.5 (split, 64 each) | biggest delta; drives their dp16xtp8 vs our dp16xtp4 |
| rollout_tensor_parallelism | 8 | 4 | our TP preference came from 2x2x1 (4 devices), untested at scale |
| hbm_utilization_vllm | 0.3 | 0.6 | inherited from our gemma4 recipe |
| CD max_decode_steps | 128 | 64 | |
| CONTINUE_DECODE_EOS_CHECK_INTERVAL | 8 | 0 | they warn of a 3x regression at their old config |
| RPA_V3_DECODE_BLOCK_SIZES | 1,1024,1,1024 | unset (knob R1_RPA; our validated tuple 1,8704,1,2176) | |
| chips_per_vm | 8 | 4 | ours matches the node's 4 chips/host; confirm which tpu7x wants |

Colocated mesh is the highest-value rung: it changes both utilization and the compiled train_step
program shape (128-way vs 64-way SPMD), which also bears on the ~25-min compile wall.
