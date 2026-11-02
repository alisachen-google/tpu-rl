# gemma4-E2B RL on the latest image: patch audit and the patch-free recipe (2026-11-02)

**Result.** Of the 37 runtime patches the gemma4-E2B RL recipes carried, **two fixes are real and go upstream as MaxText
PRs; everything else is either already in the latest image, replaced by a native config knob, or measured to make no
difference.** The final recipe — the optimized 24 s OpenMathInstruct recipe at response cap 8192 on the latest image,
p-state 6 — runs at **53.4 s per update (median)**, the same as with all 15 remaining patches (54.7 s), with
byte-identical rollouts. It contains no patch code except the KV-cache fix, which is needed until the MaxText PR merges.

Image: `maxtext-rl-ironwood:2026-09-22-head` (sha256 `4dc6ca72…`; jax 0.11.0, libtpu 0.0.44, vLLM 0.28.1rc1),
Pathways `20261028-jax_0.11.0`, tpu7x 4x4x4 (64 chips), OpenMathInstruct-2, 256 prompts × 8 samples, prefill 4096,
response cap 8192 (`max_target_length=12288`), p-state 6. Upstream checked: MaxText `647d453` (2026-10-30), tunix
`c4c5da1` (2026-10-30), tpu-inference `31879bc` (2026-11-01).

## 1. Method

1. **Is it already upstream?** Every patch was run against the latest image and against current upstream main of
   MaxText, tunix and tpu-inference (sources overlaid into the image). 22 of 37 no longer apply (their code is
   upstream or their lever is a native knob) and were removed earlier; **all 15 remaining patches still applied to
   upstream main**, i.e. none was already fixed upstream.
2. **Does it help?** Live A/B on the cluster, 5–8 updates per run, same seed so every run sees the same prompts:
   full patch set vs zero functional patches, then leave-one-out per patch, then group drops where single drops showed
   nothing, with an automatic early stop (median of timed updates > baseline × 1.15, hang, or crash). Per-update
   phase times separate generation, reward scoring and trainer + weight sync, so each patch is judged in the phase it
   touches. Robustness-only fixes (pathological inputs that 8 updates rarely produce) were measured offline on
   upstream code with worst-case inputs.

## 2. Verdicts

| patch | repo | verdict | evidence |
|---|---|---|---|
| KV-sharing alias fix (`patch_kv_alias_dedupe` → `patch_kv_scatter_pr`) | MaxText | **PR — required** | without it gemma4-E2B rollouts are gibberish: GSM8K step-0 accuracy 0.00 %; with the PR's exact change 80.82 % (GPU 80.31 ± 0.24) |
| format-regex backtracking (`patch_g4_format_guard`, re-done as atomic groups) | MaxText | **PR** | upstream regex: one degenerate 66k-char completion > 240 s (a real gemma4 run once lost 793 s in one update); fixed: 0.69 s; identical matches on 20k random strings; 48/48 unit tests pass |
| continue-decode masking (`patch_cd_masking`) | tpu-inference | **drop — native config** | `CONTINUE_DECODE_EOS_CHECK_INTERVAL=0` (run each decode window to depth) gives the same speed (median 53.2 vs 52.8 s) and identical rollouts; without patch or setting the recipe is +40 % (74.7 vs 53.3 s) |
| `patch_cd_report`, `patch_cd_noglobal` | tpu-inference | drop | leave-one-out medians 53.5 / 53.8 s vs 52.8 s, identical rollouts |
| `patch_approx_topk`, `patch_runner_eos`, `patch_vllm_sampler` | tpu-inference / tunix | drop | each alone and all three together: 53.3–53.5 s, identical rollouts (gemma4's end-of-turn stop works natively) |
| `patch_gc_sync` | tunix | drop | trainer + weight-sync phase 4.0–4.2 s with and without |
| `patch_block_size` | tunix | drop | native `vllm_block_size=128` already pins the KV block size ("Using cache_config.block_size: 128") |
| `patch_fast_extract` | MaxText | drop | upstream already rewrote extraction: 0.4 ms typical / 1.4 s worst, same as patched |
| `patch_verify_timeout` | MaxText | drop | no hang or timeout in ~100 measured updates (reward phase ≤ 2.8 s); upstream's batch timeout already recovers |
| `patch_reward_parallel` / `reward_num_workers` | tunix / MaxText | drop | neither process pool can fork inside the trainer's daemonic process on this image; scoring runs in-process at ~2 s per update either way |
| `patch_lendist`, `patch_phasetime`, `patch_log_stats` | tunix | drop | logging only (kept in the A/B runs for measurement, removed from the recipe) |
| qwen3 `mrope_section` sed in the launcher | MaxText | drop | edits `qwen3.py`; irrelevant to gemma4 |

### Step times behind the verdicts (updates 2–4, same prompts; seconds)

| run | dropped from the full set | update 2 / 3 / 4 | median | rollouts vs full |
|---|---|---|---:|---|
| full patch set (`g4u8kr`) | — | 61.8 / 48.6 / 52.8 | 52.8 | — |
| zero functional patches (`g4u8kz`, repeated as `g4u8kz2`) | all | 66.0 / 76.8 / 78.0 | 76.8 | differ |
| `g4locdmaskin` | cd_masking | 77.6 / 47.5 / 53.6 | 53.6 | identical |
| `g4locdreport` | cd_report | 63.0 / 47.9 / 53.5 | 53.5 | identical |
| `g4locdnoglob` | cd_noglobal | 77.1 / 47.1 / 53.8 | 53.8 | identical |
| `g4loapproxto` | approx_topk | 76.5 / 47.2 / 53.3 | 53.3 | identical |
| `g4lorunnereo` | runner_eos | 76.9 / 47.6 / 53.3 | 53.3 | identical |
| `g4lovllmsamp` | vllm_sampler | 62.8 / 47.7 / 53.5 | 53.5 | identical |
| `g4grpcd` | cd_masking + cd_report + cd_noglobal | 74.7 (update 1) | stopped | differ (= zero) |
| `g4grpsamp` | approx_topk + vllm_sampler + runner_eos | 75.5 / 47.4 / 53.5 | 53.5 | identical |
| `g4cdmaskonly` | cd_report + cd_noglobal (keep only cd_masking) | 63.6 / 47.5 / 53.7 | 53.7 | identical |
| `g4cdnative` | all three CD patches, `CONTINUE_DECODE_EOS_CHECK_INTERVAL=0` | 74.8 / 46.9 / 53.2 | 53.2 | identical |
| **final** (`g4final8k`, updates 2–7) | all but the KV fix, native settings | 75.4 / 47.2 / 53.1 / 53.7 / 67.2 / 53.1 | **53.4** | identical |

Update 2 is a straggler batch (one long completion): it lands at 62–78 s in every configuration, so single-update
differences there are noise; the medians and the per-phase split are what the verdicts rest on.

## 3. The final recipe

`parity/manifests/g4e2bcap8k-FINAL-latest-image-openmathinstruct-cap8192-20step.yaml` (20 updates for perf benchmarking;
build: `MINIMAL=1 NOLOG=1 EOS_INTERVAL=0 REWARD_WORKERS=0 STEPS=20 python3 parity/build_g4u8k.py`). Apply with
`parity/tools/apply-jobset.sh`. Everything is configuration:

* latest image + Pathways `20261028-jax_0.11.0`, `xla_tpu_dvfs_p_state=6`
* OpenMathInstruct-2, `rl.num_generations=8`, `batch_size=256`, `max_prefill_predict_length=4096`,
  `max_target_length=12288` / `kv_cache_buffer=12288` (response cap 8192), `max_seq_token_per_tpu=12288` (native trainer
  packing), explicit `local_sa_*` splash settings
* vLLM rollout: continue-decode with `max_decode_steps=1024`, **`CONTINUE_DECODE_EOS_CHECK_INTERVAL=0`**,
  `DP_SCHED_ROUTING=round_robin`, `APPROX_TOPK=1`, `free_kv_cache_during_weight_sync=false`, `vllm_block_size=128`,
  `hbm_utilization_vllm=0.3`, `max_num_seqs=256`, `max_num_batched_tokens=33280`
* one remaining patch: the KV-sharing scatter fix (`parity/patches/patch_kv_scatter_pr.py`, exactly the MaxText PR's
  change), to be removed when an image containing the PR is available
* `gemma_data.py` stays: it is the recipe's dataset processor, passed through MaxText's `dataset_processor_path` hook
  (configuration, not a patch)

Logs of every audit run: `parity/logs/patch_audit/` (per-run key lines and the early-stop / queue decisions).

## 4. Upstream PRs

| PR | branch (fork `alisachen-google/maxtext`) | description |
|---|---|---|
| fix(vllm): don't let untouched KV-sharing aliases overwrite the donor cache | `fix/vllm-kv-shared-alias-scatter` | `upstream_prs/maxtext_kv_shared_alias_scatter.md` |
| fix(rl): make the exact-format regex immune to catastrophic backtracking | `fix/rl-format-regex-backtracking` | `upstream_prs/maxtext_format_regex_backtracking.md` |

| Add gemma4-e2b GRPO RL recipe for TPU v7x (script + tutorial) | `feat/rl-gemma4-e2b-recipe` | `upstream_prs/maxtext_gemma4_e2b_rl_recipe.md` |

The recipe PR was validated with the upstream data template and the official `google/gemma-4-E2B-it` conversion
(`parity/manifests/g4e2bup-validation-upstream-templates-it-ckpt-8step.yaml`). Updates 2–7 took
54.4 / 54.5 / 56.5 / 56.2 / 67.4 / 58.2 s, for a median of 56.4 s, with coherent rollouts and exit 0.

Tunix has only a small LoRA GSM8K config for gemma4-e2b; the full-model recipe belongs in MaxText next to E4B/26B.

No tunix or tpu-inference PR is needed: every patch for those repos was either upstream already, replaced by a native
setting, or made no difference.

## 5. Review of the three PRs (2026-11-02)

All three branches rebased onto MaxText main `5eccd1a05` (no conflicts; main did not touch these files) and re-checked:

| PR | code review | tests | lint (repo pre-commit) | change made |
|---|---|---|---|---|
| #6081 KV-sharing scatter fix | identity check is correct: gathered entries are the same Python objects as the physical list, decoder replaces donor entries only; donor pattern in the test (sliding→13, global→14, global every 5th layer) matches `gemma4_small.GEMMA4_E2B_ATTENTION_PATTERN` and `kv_donor_layer_idx` | 17 passed with fix; the 2 new tests fail on upstream code | pylint, pyink, codespell pass | none |
| #6082 format-regex atomic groups | equivalence argument holds (if `(.+?)D` fails after the first `C` it fails after any later `C`; same for `B`); Python ≥ 3.11 feature, repo requires 3.12 | 48 passed; equivalence test passes on upstream too (so it tests equivalence, not the fix); pathological test is killed by a 30 s timeout on upstream, 0.7 s with fix | pass | none |
| #6083 gemma4-e2b recipe | `chat_template_path` was a no-op: MaxText uses it only when the tokenizer has no chat template, and `gemma-4-E2B-it` ships one (the validated run's prompts show gemma4's `<\|turn>` markers); XLA-flags comment said "tuned for v6e"; all 76 config keys exist in `base.yml`/`rl.yml`; settings the validated run set explicitly but the script omits equal their defaults | validated run g4e2bup (8 updates, exit 0) | mdformat reformatted the two tables; codespell passes | dropped `chat_template_path`, rewrote the tutorial's chat-template section, fixed the comment |
