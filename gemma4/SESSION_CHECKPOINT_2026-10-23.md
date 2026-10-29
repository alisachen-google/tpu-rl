# Session checkpoint — gemma4-E2B GRPO parity + perf (2026-10-23)

Reservation `r-3d1a2` (64 chips) deactivates 20:10 PDT. This captures proven
results, artifacts, open items, and exact resume steps so nothing is lost.

## 1. PROVEN (do not re-litigate)
**Reward + OSL parity achieved** on the **old image** (`maxtext-rl-ironwood@f7c957d0`,
TP1×DP128, `raw2` ckpt, `gemma_data.py` prompt), run `g4r55b`, 10 steps:
- **check_numbers 0.58–0.62** vs GPU verl 0.54–0.65 — parity, stable across steps.
- **OSL ~1420, truncation 2.5%, natural id-106 stop**; grad_norm ~0.44 (real learning).
- **Step ~33–34 s** (= proven best): rollout ~28.6 s (86%), dequeue ~7.8 s (async-overlapped),
  train 0.6 s, weight_sync 0.4 s.
- Row-level scorer comparison vs GPU `reward_dumps` (2205 GPU rows): identical
  `maxtext_math_reward` schema/weights; TPU format-compliance 82% vs GPU 98.5% (a real
  delta to improve; not a scorer mismatch).

**Roofline (confirmed 7.37 TB/s HBM + real MQA KV):** decode matmul floor ~1.8 s =
**~6% of rollout**; **~94% is decode-step device overhead (RPA attn + 262k-vocab sampling)
+ CD-window host turnaround**, NOT prefill (ISL only ~150 tokens). See
`GEMMA4_PERF_PROFILING_HEADROOM.md`.

## 2. ARTIFACTS (preserved on GCS / in repo)
- TB scalars (parity+perf): `gs://ubench-logs/g4r55b/tensorboard/`
- lendist (OSL/trunc per step): `gs://ubench-logs/g4r55b/lendist/` (10 steps)
- phasetime: `gs://ubench-logs/g4r55b/phasetime/` (10 steps)
- Base checkpoint (THE parity lever, intact): `gs://ubench-logs/alisachen-tpu7x/ckpt/gemma4-e2b-raw2/0/items`
- ⚠️ `g4r55b/checkpoints/actor/10/` is **empty** — no trained checkpoint persisted
  (period 100 > 10 steps). Not needed for parity/perf; to keep trained weights next
  time set `checkpoint_period<=num_batches`.
- Docs: `GEMMA4_PERF_PROFILING_HEADROOM.md`, `GPU_TPU_REWARD_ALIGNMENT.md`,
  `GEMMA4_E2B_TPU_GPU_PARITY_GUIDE.md`.

## 3. NEW-IMAGE regression (root-caused, NOT fixed)
New image `4dc6ca72` produces **garbage rollout (reward 0)** for gemma4. Ruled out
(evidence in transcript): checkpoint (identical raw2), tunix adapter, update_params
dispatch, dense weight-copy, and **sampling** (offline unit test: distributed sampler
correct at TP1/TP4). → Regression is in the **gemma4 inference FORWARD** (wrong logits
pre-sampling) — an upstream code regression between the two image dates. Not fixable by
config/prompt/knobs.
- Staged fix attempts (in `g4-e2b-head-all-in-one.yaml`): 2 vLLM knobs
  (`vllm_hf_overrides=MaxTextForCausalLM` + `maxtext_config` → approach A), 14 `local_sa_*`
  params (trainer splash-attn), `gemma_data.py`+`use_chat_template=false`. These fix the
  crash/attention but NOT the forward garbage.
- Probe staged: `patches/patch_dump_rollout_logits.py` (dumps rollout top-10 logits) —
  validated, ready to confirm the forward regression on a new-image run.

## 4. OPEN ITEMS (need a fresh reservation — slice ended)
1. **Decode xplane split** — `g4r55prof` (old image, `PHASED_PROFILING_DIR` on,
   `num_batches=3`) was launched but crash-looped on a kueue TAS webhook race
   (`vpod.kb.io` denied workers, "missing selector cloud.google.com/gke-nodepool")
   and did not reach step 2. **Fix before rerun:** add `cloud.google.com/gke-nodepool`
   nodeSelector to the worker pod spec to stop the restart-recreation denials.
2. **Top perf levers** (from the roofline, attack the 94%/decode): **int8/fp8 rollout
   weights** (halve the 9.26 GB decode read), **262k-vocab sampling efficiency**, **RPA
   tile tuning for head_dim 256**, **deeper CD window**. ❌ do NOT enable prefix caching
   (measured −6 s/step when OFF, run 33j; ISL ~150 makes prefill negligible).
3. **New-image forward-logit probe** — run `g4-e2b-head` + `patch_dump_rollout_logits.py`
   to confirm the forward regression, then file upstream.

## 5. RESUME (next reservation)
```
# re-book r-xxxxx (tpu-reservations app), then on the NAP context:
CTX=gke_cloud-tpu-shared-capacity_us-central1_bodaborg-tpu7x-nap
# profiling run (fix the worker nodeSelector first):
kubectl --context $CTX apply -f <g4r55prof.yaml with gke-nodepool selector + ns=r-xxxxx>
# → capture gs://ubench-logs/g4r55prof/phased_profile/ xplane → split the 94%
```
ADC note: kubectl needs fresh ADC via `gs://alisachen/tmp/adc.json` when `invalid_rapt`
recurs (install, delete bucket copy). **Pending: delete the lingering `adc.json` copy —
my creds lack storage-delete scope.**
