# gemma4-E2B RL Benchmark — Handoff

**Last updated:** 2026-08-30 · **Repo:** `tpu-rl` · **Report of record:** `gemma4/GEMMA4_PERF_REPORT.md` (full history + attributions)

This is a self-contained handoff for continuing the gemma4-E2B GRPO rollout-performance
campaign on TPU v7x. It states (1) the current running script and champion benchmark, and
(2) the queued experiment candidates. Read `GEMMA4_PERF_REPORT.md` for the evidence behind
every claim; this file is the actionable summary.

---

## 1. Current running script + champion benchmark

### Champion: `g4r7b` — 125.8 s / step (3.90× over 490 s baseline)

Model gemma4-E2B-it, GRPO on `nvidia/OpenMathInstruct-2` (train_1M), GSM8K eval split.
Measured over 20 steps, medians of steps ≥3 (n=17):

| Phase | Median | Notes |
|---|---|---|
| **Global step** | **125.8 s** (floor 121.6) | new champion vs g4r4 128.7 |
| Rollout (`engine_generate`) | 96.7 s | **77% of step — the target** |
| Rewards + advantage | 11.7 s | spiky (up to 47 s on hard batches) |
| Trainer + weight sync | 17.9 s | |

Config: packing image + full patch roster + **rollout tp=1 × dp64** (was dp128 pre-migration; 64 chips) + tmbs=8 + sequence packing ON (`MAX_SEQ_TOKEN_PER_TPU=8192`) + CD window 64.

### The launch script

- **Launcher:** `gemma4/gemma4_e2b_r1_launcher.sh` (drives `maxtext.trainers.post_train.rl.train_rl`)
- **Chart:** `chart/` (helm) — templates a patches-configmap; `workload_patches=true patches.enabled=true`
- **Image:** `gcr.io/supercomputer-testing/maxtext-rl-ironwood:2026-08-24-packing` (tunix 408ca1d9, vLLM 0.23.1rc1-dev, JAX 0.10.2-pathways)
- **Topology:** 4×4×4 (64 chips), `chart/values-4x4x4.yaml`
- **Staging dir used for probes:** `/mnt/disks/scratch/g4r8-stage/` (chart copy + launcher + patches + env overlays)

### Runtime patch roster (24, all self-verifying + fail-open)

Champion 16: `patch_vllm_sampler cd_proper lendist rpa_blocks cd_masking cd_report phasetime reward_parallel sharded_sampling profiler_fix gc_sync packing_knob packer_fast log_stats perf_v2 dump_completions`

Added for qwen3-transfer (8): `firsttok engine_kwargs host_extract fast_detok dp_rr_routing sync_keepkv cd_noeos flash_sa` — all present in `patches/`, all validated inside the image (apply + parse + idempotent). The last two (`cd_noeos`, `flash_sa`) default OFF via env; the rest are active or env-gated.

### Launcher env knobs (set via helm `workload.envs`)

| Knob | Effect | Champion value |
|---|---|---|
| `DL_TP` / `DL_TMBS` / `DL_PACK` | rollout tp / train micro-batch / packing tokens | 1 / 8 / 8192 |
| `DL_CD` | CD window (`max_decode_steps`) | 64 |
| `DL_STEPS` | num_batches (probe length) | 20 (10 for probes) |
| `DL_BUCKET_GAP` | `VLLM_TPU_BUCKET_PADDING_GAP` — **corruption fix** | **32768 (ADOPT)** |
| `DL_DPRR` | `DP_ROUTING=rr` round-robin routing | probe |
| `DL_KEEPKV` | `SYNC_KEEPKV` keep KV across weight sync | probe |
| `DL_ENGKW` | `RL_VLLM_ENGINE_KWARGS` (e.g. prefix caching off) | probe |
| `DL_MVPROCS` | `math_verify_num_procs` | 32 (probe 64) |
| `DL_NOEOS` | `CD_NO_EOS_REDUCE` drop per-iter cross-DP EOS reduce | probe |
| `DL_ATTN` / `DL_SAQ` / `DL_SAKV` | trainer attention / splash block q / kv | dot_product (probe: flash 1024/2048) |

### How to launch (default namespace)

```
STAGE=/mnt/disks/scratch/g4r8-stage; CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap
helm --kube-context=$CTX install <name> $STAGE -f $STAGE/values-base.yaml -f $STAGE/values-4x4x4.yaml -f $STAGE/g4r9p1-envs.yaml \
  --set-file workload_launcher=$STAGE/g4r8_launcher.sh \
  --set workload.image=gcr.io/supercomputer-testing/maxtext-rl-ironwood:2026-08-24-packing \
  --set workload_patches=true --set patches.enabled=true --set workload.configFile= \
  --set maxRestarts=50 --set queue=multislice-queue --set kueue_priority_class=medium
```
**Always** `export CLOUDSDK_AUTH_ACCESS_TOKEN=$(gcloud auth application-default print-access-token)` first (the gke-gcloud-auth-plugin path breaks on `invalid_rapt`). Reinstall only AFTER old pods fully terminate (a 20 s race causes a spurious jobset failure). Priority class must be `medium|low|very-low` (not `very-high`).

---

## THE CORRECTNESS FINDING (adopt unconditionally, gates the whole ladder)

**Prefill-bucket corruption.** Token-padding buckets were pure powers of two topping at
1,048,576; the real prefill budget (8448×64 = 540,672) rounded UP to that bucket = **94%
padding**, which corrupts each sequence's **first generated token** (same bug tianyu's qwen3
found; upstream fix tpu-inference PR #3466). Symptom: completions open by echoing the prompt
instead of answering. Measured via new `patch_firsttok` census on byte-identical prompts:

- Champion (no fix): **echo 72%** of openings.
- With `VLLM_TPU_BUCKET_PADDING_GAP=32768` (max bucket → 557,056, 3% padding): **echo 39%**.

**LENDIST is unchanged across the fix** (66.8% vs 65.5% trunc; osl 2857 vs 2811) — unlike qwen3,
gemma's corruption changed content but NOT length, so the perf ladder stays comparable and does
NOT need a full re-baseline; pre-fix runs just carry a correctness asterisk. **All future runs
must set `DL_BUCKET_GAP=32768`.**

---

## 2. Experiment candidates (queued, in priority order)

Every item has an env switch → bisectable one knob at a time. Method (from tianyu, adopted):
byte-identical step-0 prompts as the gate (LENDIST + FIRSTTOK), one knob per run, 10-step probe,
20-step confirm on median steps≥3, adopt only if >0.2 s.

### IN FLIGHT — `g4r9p1` host-bundle probe (10-step)
Champion config + `DL_BUCKET_GAP=32768 DL_DPRR=1 DL_KEEPKV=1 DL_MVPROCS=64 DL_ENGKW='{"enable_prefix_caching":false}'`.
Bundles qwen3 items 0/1/5/6/9/11/12. **Status: preempt-cycling in `default` (ubench-regression cohort
holds all 256 physical chips); maxRestarts=50; no warm PHASETIME captured yet.** Env overlay:
`gemma4/g4r9p1-envs.yaml`. Needs one ~15-min quiet window OR the priority-dev lane (below).

Bundled optimizations & mechanism:
- **item 9 dp round-robin routing** (`patch_dp_rr_routing`) — likely biggest win; stock router probes
  all 64 ranks per `add_request` (~13 s/step at dp32 on qwen3; we have dp64 = 2× exposure). Attacks
  the proven Gap-A bottleneck (head runs rollout on 1–2 of 44 cores, single Python loop).
- **item 5 vectorized EOS trim** (`patch_host_extract`) — 9.7k np.isin/window → 1 matrix op (~70 ms/win).
- **item 11 batched detokenize** (`patch_fast_detok`) — 2048 sequential decodes → 1 batch_decode (~4 s).
- **item 6 keep KV across sync** (`patch_sync_keepkv`) — skip delete/reinit KV (~2 s).
- **item 1 prefix caching off** (`patch_engine_kwargs`) — no per-request all-rank cache probes.
- **item 12** verifier procs 32→64.

### NEXT — `g4r9p2` device/trainer rung (staged, validated, NOT yet launched)
g4r9p1 winners + `DL_NOEOS=1 DL_ATTN=flash DL_SAQ=1024 DL_SAKV=2048`. Two additions separable by phase
in one run:
- **item 4 drop cross-DP EOS reduce** (`patch_cd_noeos`) — makes `any_hit_eos` static False; deletes the
  per-iteration global scalar all-reduce + carried state from the compiled window. (g4r7b device window
  ≈ 15 ms is ~half small collectives; `EOS_CHECK_INTERVAL=0` already drops it from the loop *condition*
  but XLA keeps the carried state.) → rollout device time.
- **item 8 flash/splash trainer attention** (`patch_flash_sa`) — RLConfig class-attr shim providing all
  28 sa_*/local_sa_* fields (RLConfig doesn't declare them; CLI-set is rejected). Blocks q1024/kv2048.
  Activated by `attention=flash`. → trainer time (default dot_product materializes dense [B,S,S] on
  8192-token packed rows).

### THEN — re-tests unlocked by the above
- **CD window 256** (`DL_CD=256`): tested at 128 = neutral BEFORE host fixes; tianyu's 256 won only after
  routing/host wins made per-window fixed cost the remaining term. Re-test after g4r9p1/p2 land.
- **tp re-sweep on corrected generation** (`DL_TP=4` vs 1): our tp choice predates the corruption fix.
  If it goes to tp>1, WRITE `patch_ss_padfix` first (tianyu item 3): padded rows carry top_k=0 and the
  whole-batch `all(top_k>0)` guard silently disables the vocab-sharded sampler at tp>1. (Inert at tp=1,
  which is why we haven't needed it.)

### DECLINED (with reasons) — do not re-run
- **async_scheduling=true** — VOID: changes the sampling stream under continue-decode (~29% shorter gens,
  LENDIST out of band from step 0), and no per-token win. Log: `logs/g4r8-async-void.log`.
- **window 128 alone** — neutral (rollout 99.5 vs 96.7); round-trip count is not the binding constraint.
- Meta/tianyu nulls already ruled out: bf16 candidate top-k, splash 2048/4096, remat=full, fsdp trainer
  mesh, KV page 256, static sharded-sampler predicate, async-collective XLA flags, TP2/TP1-for-qwen.

### POST-LADDER — three adoptions from Meta's gemma4 recipe (see report + roadmap)
1. **Threefry replay-exact graft** (from tpu-inference PR #3100, unmerged): regenerate Gumbel noise at
   candidate GLOBAL vocab indices inside `patch_sharded_sampling` → bit-exact vs jax.random.categorical.
   Payoff: byte-identical A/B gates for all future sampler/mesh/async experiments.
2. **Stop-token hygiene** (zero-risk launcher default): gemma eos=[1,106,50] + skip_special_tokens=True
   for free-text rewards.
3. **OPD / split-pool overlap** (tunix `AgenticGRPOLearner`): overlap rollout with train at one-step
   off-policyness; our 0.5/0.5 mesh already matches OPD topology. CHANGES THE ALGORITHM CONTRACT — run
   as an explicitly-labeled variant, gate on reward-curve equivalence, not a drop-in champion candidate.

---

## Root bottleneck (why the priorities are what they are)

Rollout is ~99% host-bound: the decode kernel sits ON the memory roofline (109 TFLOP/s ≈ 100% of the
AI≈15 ceiling; ~2% device utilization per window because the host blocks 546 ms/window on `device_get`),
and **`kubectl top` during rollout shows the head pod using 1–2 of its 44 cores** — all 128 engine
drivers share one Python process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`, "required under Pathways"). So
rollout wall ≈ total host work / 1 core. Kernel/tile work is closed; the wins are host parallelization
(routing, per-window host cost) and eventually breaking the single-process head (multi-proc engines /
multi-head sharding).

---

## Infra / operational notes

- **Cluster contention:** `default` namespace ClusterQueue owns **0** tpu quota; everything is borrowed
  from the `tpu-shared-cohort`, and `ubench-regression-tests` (owns 256, uses all) reclaims mid-flight.
  Physical = 256 chips (4× 4×4×4 slices). Only 4×4×4 and 2×2×1 node pools exist — **no 32-chip pool**, so
  can't shrink to fit.
- **Priority-dev lane (the real fix):** `priority-dev` ClusterQueue owns 51 idle chips. Submitting there
  makes our run *reclaim* rather than *be reclaimed*. Requires membership in
  `tpu-priority-dev-team@twosync.google.com` (human step; RBAC no longer user-creatable per 2026-08-19
  two-layer model). One-shot script ready: `gemma4/submit_priority_dev.sh` (RBAC preflight refuses until
  access lands). NOTE: 51 < 64, so a 4×4×4 run still borrows 13 and isn't fully immune; a 4-chip 2×2×1 is
  too small for meaningful numbers.
- **ADC:** periodically dies with `invalid_rapt`. Recovery: user runs
  `gcloud auth application-default login` on a workstation, uploads to `gs://alisachen/tmp/adc.json`;
  on the VM copy to `~/.config/gcloud/application_default_credentials.json`, verify, then **delete the
  bucket copy immediately** using an ADC token (`CLOUDSDK_AUTH_ACCESS_TOKEN=... gcloud storage rm` — plain
  rm uses the scope-limited compute SA and 403s).
- **kubectl/helm:** always pass `--context=$CTX` (a background process flips context to cmcs-live);
  export `CLOUDSDK_AUTH_ACCESS_TOKEN` first.
- **Reinstall race:** uninstall → wait for pods gone → install; otherwise the jobset fails at head start.
- **Patches configmap keeps `.py` extensions:** the launcher roster loop probes both `patch_X` and
  `patch_X.py` (fixed in commit 297bd6c) — a silent skip of all patches if this regresses. Monitor gates
  on the applied-banner count.

## Artifacts / logs
- Report: `gemma4/GEMMA4_PERF_REPORT.md`. Explainer artifact (Gap-B story): claude.ai artifact
  `57e93c97-a1da-4007-86a9-47f9976d6990`.
- Logs in `gs://alisachen/tpu_rl/ironwood/logs/`: `g4r7b-head.log` (champion), `g4r8-async-void.log`,
  `g4r8b-head.log`. Profiles under `gs://ubench-logs/alisachen-tpu7x/runs/<run>/phased_profile/`.
