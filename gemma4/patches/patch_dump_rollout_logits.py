"""Forward-logit diagnostic: dump the ROLLOUT model's logits for the first decode
positions, to isolate forward-math regressions from sampling.

WHY. Bisection + an offline unit test have ruled out the checkpoint, the tunix
adapter, the update_params dispatch, the (dense) weight copy, AND the distributed
sampler (unit-tested: it picks correct top-region tokens given correct logits).
That leaves the gemma4 inference FORWARD: if the rollout produces garbage tokens
while sampling is correct, the LOGITS fed to sampling must already be wrong. This
patch prints those logits directly.

WHAT. Wraps tpu_inference.layers.jax.sample.sampling.sample(). On the first
DUMP_LOGITS_N calls (decode steps) it logs, for request row 0:
  - the argmax token id (what greedy would pick),
  - the top-10 token ids and their logit values,
  - basic logit stats (max/mean/any-nan).
A HEALTHY forward on a math prompt puts sensible tokens on top with a clear
margin; a BROKEN forward shows flat/garbage logits (random/`<unused>` ids on top,
tiny margins, or NaNs). Compared against the old image's known-good behaviour this
is decisive for "wrong logits (forward)" vs "wrong token mapping (sampling)".

Read-only (does not change sampling); fail-open; gated on env DUMP_LOGITS_N
(default 3, set 0 to disable). Prints:
    DUMP-LOGITS PATCH: APPLIED | already applied | FAILED <reason>
and at runtime: ROLLOUT-LOGITS call=K argmax=.. top10_ids=[..] top10_vals=[..] ..
"""
import os
import sys

P = "/usr/local/lib/python3.12/site-packages/tpu_inference/layers/jax/sample/sampling.py"


def fail(msg):
  print(f"DUMP-LOGITS PATCH: FAILED {msg}", flush=True)
  sys.exit(0)


if os.environ.get("DUMP_LOGITS_N", "3") == "0":
  print("DUMP-LOGITS PATCH: skipped (DUMP_LOGITS_N=0)", flush=True)
  sys.exit(0)

if not os.path.exists(P):
  fail(f"{P} not found")
src = open(P, encoding="utf-8").read()
if "_DUMP_LOGITS" in src:
  print("DUMP-LOGITS PATCH: already applied", flush=True)
  sys.exit(0)

anchor = "def sample(\n"
if anchor not in src:
  fail("sample() def anchor not found")

# Append a wrapper at EOF that rebinds `sample` to log logits[0] then delegate.
appendix = '''

# --- _DUMP_LOGITS: read-only forward-logit diagnostic -----------------------------
import os as _dl_os
import numpy as _dl_np
import jax as _dl_jax

_DL_N = int(_dl_os.environ.get("DUMP_LOGITS_N", "3"))
_DL_STATE = {"calls": 0}
_dl_orig_sample = sample  # the module-level sample defined above


def sample(rng, mesh, logits, tpu_sampling_metadata, allow_distributed_sampling=True):  # noqa: F811
  if _DL_STATE["calls"] < _DL_N:
    try:
      # host-copy row 0 of the raw logits (block=fine, diagnostic only)
      l0 = _dl_np.asarray(_dl_jax.device_get(logits[0])).astype("float32")
      k = 10
      top = _dl_np.argsort(-l0)[:k]
      vals = [round(float(l0[i]), 3) for i in top]
      argmax = int(top[0])
      print("ROLLOUT-LOGITS call=%d argmax=%d top10_ids=%s top10_vals=%s "
            "logit_max=%.3f logit_mean=%.3f n_nan=%d vocab=%d"
            % (_DL_STATE["calls"], argmax, [int(t) for t in top], vals,
               float(_dl_np.nanmax(l0)), float(_dl_np.mean(l0)),
               int(_dl_np.isnan(l0).sum()), l0.shape[-1]), flush=True)
    except Exception as _e:  # pragma: no cover
      print("ROLLOUT-LOGITS dump error: %r" % (_e,), flush=True)
    _DL_STATE["calls"] += 1
  return _dl_orig_sample(rng, mesh, logits, tpu_sampling_metadata,
                         allow_distributed_sampling=allow_distributed_sampling)
# --- end _DUMP_LOGITS -------------------------------------------------------------
'''

src2 = src + appendix
try:
  import ast
  ast.parse(src2)
except SyntaxError as e:
  fail(f"patched source does not parse: {e}")
open(P, "w", encoding="utf-8").write(src2)
print(f"DUMP-LOGITS PATCH: APPLIED (DUMP_LOGITS_N={os.environ.get('DUMP_LOGITS_N', '3')})", flush=True)
