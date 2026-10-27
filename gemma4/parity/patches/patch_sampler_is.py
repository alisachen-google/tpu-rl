"""SAMPLER-IS (TIS parity) for tunix 0.1.8: enable token-level truncated importance sampling
on the STANDARD (offline-rollout) GrpoLearner by config, matching the frozen GPU recipe:

    loss = -A * w * logπ_θ   (global token-mean; REINFORCE, no PPO-clip effect, no KL)
    w    = min(exp(clamp(logπ_actor.detach() - logπ_sampler, -20, 20)), 3)   # detached, masked

tunix 0.1.8's standard grpo_learner ALREADY computes sampler_is_weights with exactly this
formula (rl/grpo/grpo_learner.py: log_ratio = trainer - rollout, clip +/-20, min(exp, threshold),
stop_gradient, masked) and applies them in grpo_loss_fn -- gated on algo_config.sampler_is=="token".
It also needs the sampler's rollout logprobs (RolloutConfig.return_logprobs=True). The GPU-parity
container is tunix 0.1.8 (image @sha256:f7c957d0...), NOT 0.1.7. So NO tunix code patch is needed:
MaxText's train_rl just has to (a) pass sampler_is / sampler_is_threshold into GrpoConfig and
(b) enable return_logprobs on the offline RolloutConfig. This patch does exactly that.

With num_iterations=1 and sampler_is=="token", old_per_token_logps is set to the actor recompute
so the PPO ratio is ~1 (clip inert); the sampler correction rides entirely on w -> -A*w*logπ_θ.

FAIL-LOUD (exit!=0 so `set -e` kills the pod rather than training without TIS), idempotent,
ast-verified. Prints: TIS-PARITY PATCH: APPLIED | already patched | FAILED <reason>.
Runtime self-check: grpo logs sampler_is/weight_min < 1.0 when TIS is active (==1.0 => inert).
"""

import ast
import os
import sys

THRESHOLD = "3.0"  # GPU recipe caps w at 3 (upper only); tunix default is 2.0


def fail(msg):
    print(f"TIS-PARITY PATCH: FAILED {msg}", flush=True)
    sys.exit(1)


def find_train_rl():
    try:
        import maxtext
    except Exception as e:
        fail(f"cannot import maxtext: {e}")
    root = os.path.dirname(maxtext.__file__)
    p = os.path.join(root, "trainers/post_train/rl/train_rl.py")
    if not os.path.exists(p):
        fail(f"train_rl.py not found at {p}")
    return p


TR = find_train_rl()
src = open(TR, encoding="utf-8").read()

if "_TIS_PARITY" in src:
    print("TIS-PARITY PATCH: already patched", flush=True)
    sys.exit(0)

# ---- A) forward sampler_is / sampler_is_threshold into the STANDARD GrpoConfig ----
A_ANCHOR = """    grpo_config = GrpoConfig(
        num_generations=trainer_config.rl.num_generations,
        num_iterations=trainer_config.rl.num_iterations,
        beta=trainer_config.rl.grpo_beta,
        epsilon=trainer_config.rl.grpo_epsilon,
        loss_algo=trainer_config.rl.loss_algo,
        loss_agg_mode=trainer_config.rl.loss_agg_mode,
    )"""
A_REPLACE = """    grpo_config = GrpoConfig(
        num_generations=trainer_config.rl.num_generations,
        num_iterations=trainer_config.rl.num_iterations,
        beta=trainer_config.rl.grpo_beta,
        epsilon=trainer_config.rl.grpo_epsilon,
        loss_algo=trainer_config.rl.loss_algo,
        loss_agg_mode=trainer_config.rl.loss_agg_mode,
        sampler_is="token",  # _TIS_PARITY: token-level truncated IS (GPU frozen recipe)
        sampler_is_threshold=%s,  # _TIS_PARITY: GPU caps w at 3 (upper only)
    )""" % THRESHOLD

# ---- B) enable sampler logprobs on the offline RolloutConfig ----
B_ANCHOR = (
    '          **({"return_logprobs": True} if trainer_config.rl.use_agentic_rollout else {}),'
)
B_REPLACE = (
    "          return_logprobs=True,  # _TIS_PARITY: sampler per-token logprobs for token TIS"
)

# ---- C) EVAL COVERAGE: score the FULL GPU eval set, not 10/11 batches. The eval dataset is
# batched with drop_remainder=True; 1319 (prime) GSM8K test Qs @ bs=128 -> floor=10 batches=1280,
# silently dropping the last 39 and making the accuracy denominator differ from the GPU's 1319.
# The eval generate path is request-based vLLM (one request per prompt, no fixed-count/DP-divisibility
# requirement -- verified), so a ragged final batch is safe. Flip ONLY the eval batch (keyed on
# eval_batch_size_for_eval; the TRAIN batch keyed on trainer_config.batch_size is untouched).
C_ANCHOR = "    test_dataset = test_dataset.to_iter_dataset().batch(eval_batch_size_for_eval, drop_remainder=True)"
C_REPLACE = "    test_dataset = test_dataset.to_iter_dataset().batch(eval_batch_size_for_eval, drop_remainder=False)  # _TIS_PARITY: full 1319 coverage vs GPU"

# ---- F) decouple ROLLOUT dtype from trainer weight_dtype. train_rl sets the vLLM rollout
# "dtype" = trainer_config.weight_dtype.value; with weight_dtype=float32 (fp32 master) that makes
# the rollout fp32 -> its KV cache is fp32 while attention k/v are bf16 -> RPA kernel ValueError
# (kv_cache.dtype float32 != k/v bfloat16). Force the rollout to bf16 regardless of the trainer
# master dtype (GPU recipe: fp32 master+Adam, BF16 compute AND bf16 rollout). ----
F_ANCHOR = '          "dtype": trainer_config.weight_dtype.value,'
F_REPLACE = '          "dtype": "bfloat16",  # _TIS_PARITY: rollout bf16 (decoupled from fp32 trainer master)'

# ---- H) tensorboard metrics to LOCAL disk. tensorboardX's GCSRecordWriter flushes the event file
# from two threads (EventFileWriter background thread + main-thread flush) each with a GCS
# ifGenerationMatch precondition; under the heavy eval(+checkpoint) traffic at step 40 they raced
# into `412 PreconditionFailed` in the MAIN thread (rl_learner._prepare_data) and killed g4pt1.
# The tensorboard RL metrics are not used for parity (stdout is), so write them locally. ----
H_ANCHOR = (
    "  metrics_logging_options = metrics_logger.MetricsLoggerOptions(\n"
    "      log_dir=trainer_config.tensorboard_dir, flush_every_n_steps=trainer_config.log_period\n"
    "  )"
)
H_REPLACE = (
    "  metrics_logging_options = metrics_logger.MetricsLoggerOptions(\n"
    "      log_dir=(os.makedirs(\"/tmp/tb_local\", exist_ok=True) or \"/tmp/tb_local\"),  # _TIS_PARITY: local TB (GCS 412 race)\n"
    "      flush_every_n_steps=trainer_config.log_period,\n"
    "  )\n"
    "  try:  # _TIS_PARITY: mirror every logged RL metric (loss/entropy/pg_clipfrac/kl/lr/...) to stdout\n"
    "    _tp_ML = metrics_logger.MetricsLogger\n"
    "    if not getattr(_tp_ML, '_tp_stdout', False):\n"
    "      _tp_ML._tp_orig_log = _tp_ML.log\n"
    "      def _tp_fmt(x):\n"
    "        try:\n"
    "          return '%.6g' % float(x)\n"
    "        except Exception:\n"
    "          return repr(x)[:80]\n"
    "      def _tp_log(self, *a, **k):\n"
    "        try:\n"
    "          print('RLMETRIC ' + ' '.join([_tp_fmt(x) for x in a] + ['%s=%s' % (kk, _tp_fmt(vv)) for kk, vv in k.items()]), flush=True)\n"
    "        except Exception:\n"
    "          pass\n"
    "        return _tp_ML._tp_orig_log(self, *a, **k)\n"
    "      _tp_ML.log = _tp_log\n"
    "      _tp_ML._tp_stdout = True\n"
    "  except Exception as _tp_e:\n"
    "    print('RLMETRIC-WRAP skipped: %r' % (_tp_e,), flush=True)"
)

for anchor, label in ((A_ANCHOR, "GrpoConfig"), (B_ANCHOR, "return_logprobs"),
                      (C_ANCHOR, "eval_coverage"), (F_ANCHOR, "rollout_dtype"),
                      (H_ANCHOR, "tb_local")):
    n = src.count(anchor)
    if n != 1:
        fail(f"{label}: anchor count {n} != 1 in {TR}")
if "\nimport os\n" not in src and not src.startswith("import os\n"):
    fail("train_rl.py has no `import os` (needed by tb_local edit)")

new = (
    src.replace(A_ANCHOR, A_REPLACE, 1)
    .replace(B_ANCHOR, B_REPLACE, 1)
    .replace(C_ANCHOR, C_REPLACE, 1)
    .replace(F_ANCHOR, F_REPLACE, 1)
    .replace(H_ANCHOR, H_REPLACE, 1)
)
try:
    ast.parse(new)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")

open(TR, "w", encoding="utf-8").write(new)
v = open(TR, encoding="utf-8").read()
if (
    'sampler_is="token"' not in v
    or "return_logprobs=True,  # _TIS_PARITY" not in v
    or "drop_remainder=False)  # _TIS_PARITY" not in v
):
    fail("verification markers missing after write")
try:
    ast.parse(v)
except SyntaxError as e:
    fail(f"reparse failed: {e}")

# ---- D) stdout-print TIS weights in the standard learner (tensorboard RL metrics log as NaN;
#         prove TIS is biting via stdout: weight_min<1 => active, line absent => inert) ----
try:
    import tunix as _tnx
    _GL = os.path.join(os.path.dirname(_tnx.__file__), "rl/grpo/grpo_learner.py")
except Exception as e:
    fail(f"cannot locate tunix grpo_learner: {e}")
gsrc = open(_GL, encoding="utf-8").read()
if "_TIS_STDOUT" in gsrc:
    print("TIS-PARITY PATCH: grpo_learner stdout already patched", flush=True)
else:
    # E) unconditional debug print right before the sampler_is block, to pinpoint which
    #    guard fails (sampler_is value / rollout logprobs / trainer logprobs).
    E_ANCHOR = (
        '    if (\n'
        '        self.algo_config.sampler_is == "token"\n'
        '        and rollout_per_token_logps is not None\n'
        '        and trainer_per_token_logps is not None\n'
        '    ):'
    )
    E_REPLACE = (
        '    print("TIS-DEBUG sampler_is=%r use_rollout_logps=%r roll_lp_set=%s train_lp_set=%s "\n'
        '          "ro_logprobs_set=%s"\n'
        '          % (self.algo_config.sampler_is, self.algo_config.use_rollout_logps,\n'
        '             rollout_per_token_logps is not None, trainer_per_token_logps is not None,\n'
        '             (getattr(rollout_output, "logprobs", None) is not None)), flush=True)  # _TIS_STDOUT\n'
        + E_ANCHOR
    )
    _ne = gsrc.count(E_ANCHOR)
    if _ne != 1:
        fail(f"grpo_learner TIS-DEBUG anchor count {_ne} != 1 in {_GL}")
    gsrc = gsrc.replace(E_ANCHOR, E_REPLACE, 1)

    D_ANCHOR = (
        '      self.rl_engine.buffer_metrics(\n'
        '          {\n'
        '              "sampler_is/weight_mean": (is_mean, np.mean),'
    )
    D_REPLACE = (
        '      _tis_wmin = float(jnp.min(jnp.where(asst_mask_f > 0, sampler_is_weights, 1.0)))  # _TIS_STDOUT\n'
        '      _tis_w = jnp.exp(log_ratio) * asst_mask_f  # unclipped token weights (GPU ESS / k3 definitions)\n'
        '      _tis_ess = float(_tis_w.sum() ** 2 / jnp.maximum((_tis_w ** 2).sum() * mask_sum, 1e-12))\n'
        '      _tis_d = -log_ratio  # log(pi_sampler) - log(pi_trainer)\n'
        '      _tis_k3 = float(((jnp.exp(_tis_d) - _tis_d - 1.0) * asst_mask_f).sum() / mask_sum)\n'
        '      _tis_lrm = float((log_ratio * asst_mask_f).sum() / mask_sum)\n'
        '      _tis_lra = float((jnp.abs(log_ratio) * asst_mask_f).sum() / mask_sum)\n'
        '      print("TIS-ACTIVE weight_mean=%.5f weight_min=%.5f weight_max=%.5f frac_clipped=%.6f "\n'
        '            "k3_kl=%.4e ess=%.6f logratio_mean=%.4e logratio_absmean=%.4e ntok=%d"\n'
        '            % (is_mean, _tis_wmin, is_max, frac_clipped, _tis_k3, _tis_ess, _tis_lrm, _tis_lra, int(mask_sum)), flush=True)\n'
        '      self.rl_engine.buffer_metrics(\n'
        '          {\n'
        '              "sampler_is/weight_mean": (is_mean, np.mean),'
    )
    n = gsrc.count(D_ANCHOR)
    if n != 1:
        fail(f"grpo_learner stdout anchor count {n} != 1 in {_GL}")
    gsrc = gsrc.replace(D_ANCHOR, D_REPLACE, 1)
    try:
        ast.parse(gsrc)
    except SyntaxError as e:
        fail(f"grpo_learner stdout patch does not parse: {e}")
    open(_GL, "w", encoding="utf-8").write(gsrc)

# ---- G) EVAL METRIC PARITY: print scorer-v2 `acc` for the eval set. MaxText's eval `accuracy`
# (corr/total) comes from its OWN check_correctness (own extraction + math_verify), NOT scorer-v2;
# `mean_reward` is the mean v2 score (1.0/0.1/0.0 exclusive) so acc_v2 is only bounded by it
# ((mean-0.1)/0.9 <= acc <= mean). The GPU curve reports v2 acc (step0 80.31, step20 91.03,
# step250 92.42). Accumulate reward_parity_boxed.compute_acc per (prompt,response) inside
# evaluate() and log one `EVAL-V2ACC total=.. v2_acc=..%` line before the summary line. ----
EV = os.path.join(os.path.dirname(TR), "evaluate_rl.py")
if not os.path.exists(EV):
    fail(f"evaluate_rl.py not found at {EV}")
esrc = open(EV, encoding="utf-8").read()
if "_TIS_PARITY" in esrc:
    print("TIS-PARITY PATCH: evaluate_rl already patched", flush=True)
else:
    G1_ANCHOR = (
        "  reward_sum = 0.0\n"
        "  reward_count = 0  # number of (prompt, sampled response) pairs scored\n"
    )
    G1_REPLACE = G1_ANCHOR + "  _v2_acc_sum, _v2_acc_n, _v2_rows = 0.0, 0, []  # _TIS_PARITY: scorer-v2 acc + per-question rows\n"
    G2_ANCHOR = (
        "        row_sum, row_count = _compute_row_reward(reward_fns, prompt, question, responses, answer, total)\n"
        "        reward_sum += row_sum\n"
        "        reward_count += row_count\n"
    )
    # Per-question record: what the GPU side's per-question eval dump carries (gold, extracted
    # boxed answer, v2 score/acc/fmt, length) so TPU-vs-GPU can be diffed question by question.
    G2_REPLACE = G2_ANCHOR + r'''        try:  # _TIS_PARITY: scorer-v2 acc + per-question record (GPU reference curve's metric)
          import sys as _tp_sys, hashlib as _tp_hl
          if "/workload/patches" not in _tp_sys.path:
            _tp_sys.path.insert(0, "/workload/patches")
          import reward_parity_boxed as _tp_rpb
          _tp_gold = _tp_rpb._gold_str(answer)
          for _tp_r in responses:
            _tp_o = _tp_rpb._v2.compute_score(data_source=_tp_rpb._DATA_SOURCE, solution_str=_tp_r, ground_truth=_tp_gold, extra_info=None)
            _v2_acc_sum += float(_tp_o["acc"])
            _v2_acc_n += 1
            _v2_rows.append({"i": total, "qhash": _tp_hl.sha1(str(question).strip().encode()).hexdigest()[:12],
                             "gold": _tp_gold, "boxed": _tp_rpb._v2.v1.extract_boxed(_tp_r),
                             "score": float(_tp_o["score"]), "acc": float(_tp_o["acc"]), "fmt": float(_tp_o.get("fmt", 0.0)),
                             "n_chars": len(_tp_r), "tail": _tp_r[-80:]})
        except Exception as _tp_e:  # pylint: disable=broad-exception-caught
          max_logging.log(f"[eval-v2acc] failed on row {total}: {_tp_e!r}")
'''
    G3_ANCHOR = "  # Prepare return values\n"
    # Per-eval breakdown line + JSONL dump at <base_output_directory>/<run_name>/evals/eval_K.jsonl
    # (K = eval call index: 0 = step-0, then every eval_interval). Fail-open: never breaks the eval.
    G3_REPLACE = r'''  try:  # _TIS_PARITY: per-eval breakdown + per-question JSONL dump (for GPU comparison)
    import json as _tp_json
    from etils import epath as _tp_epath
    _tp_n = max(_v2_acc_n, 1)
    _tp_acc = sum(r["acc"] for r in _v2_rows)
    _tp_bw = sum(1 for r in _v2_rows if r["acc"] == 0.0 and r["boxed"] not in (None, ""))
    _tp_nb = sum(1 for r in _v2_rows if r["boxed"] in (None, ""))
    _tp_mc = sum(r["n_chars"] for r in _v2_rows) / _tp_n
    _tp_k = getattr(evaluate, "_tp_k", 0)
    evaluate._tp_k = _tp_k + 1
    try:  # actual RL update step (restart-safe: a resumed process restarts the k counter)
      _tp_step = int(rl_cluster.actor_trainer.train_steps)
    except Exception:  # pylint: disable=broad-exception-caught
      _tp_step = 20 * _tp_k
    _tp_dir = f"{str(getattr(tmvp_config, 'base_output_directory', '/tmp')).rstrip('/')}/{getattr(tmvp_config, 'run_name', 'run')}/evals"
    _tp_path = f"{_tp_dir}/eval_{_tp_step:03d}.jsonl"
    _tp_epath.Path(_tp_dir).mkdir(parents=True, exist_ok=True)
    _tp_epath.Path(_tp_path).write_text("\n".join(_tp_json.dumps(r) for r in _v2_rows) + "\n")
    max_logging.log(f"EVAL-BREAKDOWN step={_tp_step} k={_tp_k} total={_v2_acc_n} acc={_tp_acc / _tp_n * 100:.2f}% boxed_wrong={_tp_bw / _tp_n * 100:.2f}% nobox={_tp_nb / _tp_n * 100:.2f}% mean_chars={_tp_mc:.0f} dump={_tp_path}")
  except Exception as _tp_e:  # pylint: disable=broad-exception-caught
    max_logging.log(f"[eval-dump] failed: {_tp_e!r}")
  max_logging.log(  # _TIS_PARITY
      f"EVAL-V2ACC total={_v2_acc_n} v2_acc={(_v2_acc_sum / _v2_acc_n * 100 if _v2_acc_n else 0.0):.4f}% "
      f"mean_reward={(reward_sum / reward_count if (use_reward and reward_count > 0) else 0.0):.4f}"
  )
''' + G3_ANCHOR
    for anchor, label in ((G1_ANCHOR, "v2acc_init"), (G2_ANCHOR, "v2acc_accum"), (G3_ANCHOR, "v2acc_print")):
        n = esrc.count(anchor)
        if n != 1:
            fail(f"evaluate_rl {label}: anchor count {n} != 1 in {EV}")
    enew = esrc.replace(G1_ANCHOR, G1_REPLACE, 1).replace(G2_ANCHOR, G2_REPLACE, 1).replace(G3_ANCHOR, G3_REPLACE, 1)
    try:
        ast.parse(enew)
    except SyntaxError as e:
        fail(f"patched evaluate_rl does not parse: {e}")
    open(EV, "w", encoding="utf-8").write(enew)
    if "EVAL-V2ACC" not in open(EV, encoding="utf-8").read():
        fail("evaluate_rl verification marker missing after write")

# ---- I) RESUMABLE CHECKPOINTS. Every restart after a mid-run save died in tunix
# checkpoint_manager.maybe_restore: orbax v1 NoEntryError "incompatible handler" whose ROOT cause is
# `ValueError: Unsupported dtype: key<fry>` -- nnx.state(model) includes the model's RNG stream
# (typed jax.random keys) and the Pathways persistence array handler cannot describe that dtype on
# load. Save and restore `model_params` WITHOUT RngState and allow a partial load (checkpoints already
# written with keys still restore). Verified in the pinned image on CPU (orbax 0.12.1 / flax 0.12.8). ----
try:
    _CM = os.path.join(os.path.dirname(_tnx.__file__), "sft/checkpoint_manager.py")
except Exception as e:  # pylint: disable=broad-exception-caught
    fail(f"cannot locate tunix checkpoint_manager: {e}")
csrc = open(_CM, encoding="utf-8").read()
if "_TIS_PARITY" in csrc:
    print("TIS-PARITY PATCH: checkpoint_manager already patched", flush=True)
else:
    I1_ANCHOR = "      params = nnx.state(model)\n"
    I1_REPLACE = "      params = nnx.state(model, nnx.Not(nnx.RngState))  # _TIS_PARITY: PRNG keys (key<fry>) are not loadable under Pathways\n"
    I2_ANCHOR = (
        "      model_params_state = nnx.state(model)\n"
        "      load_ctx = self._context\n"
    )
    I2_REPLACE = (
        "      model_params_state = nnx.state(model, nnx.Not(nnx.RngState))  # _TIS_PARITY: see save()\n"
        "      load_ctx = ocp.Context(self._context)\n"
        "      load_ctx.pytree.loading.partial_load = True  # _TIS_PARITY: checkpoint may still carry the RNG entries\n"
    )
    for anchor, label in ((I1_ANCHOR, "ckpt_save"), (I2_ANCHOR, "ckpt_restore")):
        n = csrc.count(anchor)
        if n != 1:
            fail(f"checkpoint_manager {label}: anchor count {n} != 1 in {_CM}")
    cnew = csrc.replace(I1_ANCHOR, I1_REPLACE, 1).replace(I2_ANCHOR, I2_REPLACE, 1)
    try:
        ast.parse(cnew)
    except SyntaxError as e:
        fail(f"patched checkpoint_manager does not parse: {e}")
    open(_CM, "w", encoding="utf-8").write(cnew)

print(
    "TIS-PARITY PATCH: APPLIED (train_rl: sampler_is=token thr=%s, return_logprobs=True, "
    "eval full-coverage drop_remainder=False, rollout bf16, RLMETRIC stdout mirror; grpo_learner TIS-ACTIVE(+k3/ess) stdout; "
    "evaluate_rl EVAL-V2ACC step-named dumps; checkpoint_manager RNG-free resumable ckpt)" % THRESHOLD,
    flush=True,
)
