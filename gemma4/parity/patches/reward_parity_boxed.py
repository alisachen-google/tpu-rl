"""Reward adapter: expose the frozen GPU scorer-v2 (boxed_math_reward_v2.compute_score)
through the tunix/MaxText reward_fn interface for the gemma4-E2B GSM8K accuracy-parity run.

CONTRACT (TPU_INSTRUCTIONS.md sec 3): scorer-v2 is the SINGLE reward for both training and
evaluation. Exclusive levels: correct boxed answer = 1.0, parseable-but-wrong box = 0.1,
no/unparseable box = 0.0. It is NOT accuracy + 0.1*format. acc (1/0) is the separate
evaluation-accuracy signal, exposed as `compute_acc` for the eval metric so the parity curve
measures the same quantity as the GPU reference (92.42% = fraction with acc==1), not mean score.

The scorer files (boxed_math_reward_v2.py + its hash-checked boxed_math_reward.py dependency)
are mounted alongside this file at /workload/patches and imported by name. Hashes are verified
against TPU_INSTRUCTIONS.md sec 2 at preflight: v2=933abb..., v1=a743cf....

tunix calls reward_fn(prompts, completions, **kwargs) on CHUNKS of the batch and concatenates
in order (see the straggler's reward-parallel patch); per-example independence makes chunking
exact. The ground-truth column arrives as a per-example kwarg; we accept the common names.
"""

import json
import os
import sys

# The scorer modules are mounted next to this file; make that dir importable in every worker.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import boxed_math_reward_v2 as _v2  # noqa: E402  (mounted scorer, hash-verified)

_DATA_SOURCE = "gsm8k_boxed_test"  # value is informational to the scorer; kept stable


def _ground_truths(n, answer, kwargs):
    """Resolve the per-example ground-truth list from whatever column name the data
    processor emitted. Must be length n (one per completion). Raises on mismatch rather
    than silently scoring against the wrong gold."""
    gts = answer
    if gts is None:
        for key in ("ground_truth", "gts", "answers", "target", "gt"):
            if key in kwargs and kwargs[key] is not None:
                gts = kwargs[key]
                break
    if gts is None:
        raise ValueError("reward_parity_boxed: no ground-truth kwarg found "
                         f"(saw keys: {sorted(kwargs)})")
    if not isinstance(gts, (list, tuple)):
        gts = [gts] * n
    if len(gts) != n:
        raise ValueError(f"reward_parity_boxed: gt length {len(gts)} != completions {n}")
    return gts


def _gold_str(gt):
    """The data processor emits each gold as a JSON list string, e.g. '["18"]' (the harness
    convention evaluate_rl.py json.loads). Parse it back to the bare gold the scorer expects;
    accept a plain string or number too, for robustness."""
    if isinstance(gt, (int, float)):
        return str(gt)
    s = str(gt)
    try:
        parsed = json.loads(s)
        if isinstance(parsed, (list, tuple)) and parsed:
            return str(parsed[0])
        if isinstance(parsed, (str, int, float)):
            return str(parsed)
    except Exception:
        pass
    return s


def _score_one(completion, gt, want):
    """Return the requested field ('score' training reward, or 'acc' eval accuracy) for one
    (completion, gold) pair. Fail-closed to 0.0 so a scorer exception can never inflate the
    parity number."""
    try:
        out = _v2.compute_score(
            data_source=_DATA_SOURCE,
            solution_str=completion,
            ground_truth=_gold_str(gt),
            extra_info=None,
        )
        return float(out[want])
    except Exception as e:  # pragma: no cover - defensive
        print(f"reward_parity_boxed: scorer error ({e!r}); scoring 0.0", flush=True)
        return 0.0


def compute_score(prompts, completions, answer=None, **kwargs):
    """TRAINING reward: scorer-v2 score in {1.0, 0.1, 0.0}. One float per completion."""
    gts = _ground_truths(len(completions), answer, kwargs)
    return [_score_one(c, gts[i], "score") for i, c in enumerate(completions)]


def compute_acc(prompts, completions, answer=None, **kwargs):
    """EVALUATION accuracy: scorer-v2 acc in {1.0, 0.0} (correct boxed answer). The quantity
    the GPU reference curve reports (step-250 mean 92.42%)."""
    gts = _ground_truths(len(completions), answer, kwargs)
    return [_score_one(c, gts[i], "acc") for i, c in enumerate(completions)]
