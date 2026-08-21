"""Apply the PROPER continue-decode fix (PR #3374) to the installed tpu_inference.

Mirrors https://github.com/vllm-project/tpu-inference/pull/3374 exactly:

  (a) models/common/model_loader.py -- build an options-free twin of run_model and
      expose it as `step_fn_no_options` on the model object, mirroring the torchax
      path in models/vllm/vllm_model_wrapper.py.
  (b) runner/decode_loop.py -- build the fused-decode loop jit lazily from
      get_step_fn_compiler_options() instead of a hardcoded 3-entry literal, so the
      fused program keeps all 5 options (incl. the SparseCore offload thresholds).

Every edit asserts its anchor and re-verifies after writing. Prints exactly one of:
    CD-PROPER PATCH: APPLIED
    CD-PROPER PATCH: FAILED <reason>
A FAILED line means the run proves nothing -- check it before reading any result.
"""

import ast
import os
import sys

ROOT = "/usr/local/lib/python3.12/site-packages/tpu_inference"
ML = os.path.join(ROOT, "models/common/model_loader.py")
DL = os.path.join(ROOT, "runner/decode_loop.py")


def fail(msg):
    print(f"CD-PROPER PATCH: FAILED {msg}")
    sys.exit(0)  # non-fatal: let the run start so the failure is visible in logs


def sub(text, old, new, what):
    if text.count(old) != 1:
        fail(f"[{what}] anchor found {text.count(old)} times, expected 1")
    return text.replace(old, new)


# ---------------------------------------------------------------- (a) model_loader
if not os.path.exists(ML):
    fail(f"{ML} not found")
ml = open(ML, encoding="utf-8").read()

if "step_fn_no_options" in ml and "run_model_no_options" in ml:
    print("CD-PROPER PATCH: model_loader already patched")
else:
    if "\nimport functools\n" not in ml:
        ml = sub(ml, "from typing import Any, Optional\n",
                 "import functools\nfrom typing import Any, Optional\n",
                 "ml-import")

    ml = sub(
        ml,
        """    @jax.jit(
        out_shardings=(
            kv_cache_sharding,
            hidden_states_sharding,
            hidden_states_sharding,  # aux hidden states
            None,  # expert ids
        ),
        donate_argnums=1,  # 0 is state_leaves, 1 is kv_cache
        static_argnums=(
            6, 9, 10
        ),  # 6 is layer_name_to_kvcache_index, 9 is is_first_rank, 10 is is_last_rank
        compiler_options=get_step_fn_compiler_options(),
    )
    def run_model(state_leaves, *args):
        state = jax.tree_util.tree_unflatten(_state_treedef, state_leaves)
        model = nnx.merge(graphdef, state)
        return model(*args)
""",
        """    def run_model_impl(state_leaves, *args):
        state = jax.tree_util.tree_unflatten(_state_treedef, state_leaves)
        model = nnx.merge(graphdef, state)
        return model(*args)

    run_model_jit = functools.partial(
        jax.jit,
        out_shardings=(
            kv_cache_sharding,
            hidden_states_sharding,
            hidden_states_sharding,  # aux hidden states
            None,  # expert ids
        ),
        donate_argnums=1,  # 0 is state_leaves, 1 is kv_cache
        static_argnums=(
            6, 9, 10
        ),  # 6 is layer_name_to_kvcache_index, 9 is is_first_rank, 10 is is_last_rank
    )

    run_model_no_options = run_model_jit(run_model_impl)

    run_model = run_model_jit(
        run_model_impl,
        compiler_options=get_step_fn_compiler_options(),
    )
""",
        "ml-run_model")

    ml = sub(
        ml,
        """    def wrapped_model_fn(*args, **kwargs):
        if not model_supports_spec_step:
            kwargs.pop("spec_step_idx", None)
        kwargs.pop("shared_attention_metadata", None)
        return jitted_model_fn(*args, **kwargs)
""",
        """    def wrapped_model_fn(*args, **kwargs):
        if not model_supports_spec_step:
            kwargs.pop("spec_step_idx", None)
        kwargs.pop("shared_attention_metadata", None)
        return jitted_model_fn(*args, **kwargs)

    jitted_model_fn_no_options = (run_draft_model
                                  if is_draft_model else run_model_no_options)

    def wrapped_model_fn_no_options(*args, **kwargs):
        if not model_supports_spec_step:
            kwargs.pop("spec_step_idx", None)
        kwargs.pop("shared_attention_metadata", None)
        return jitted_model_fn_no_options(*args, **kwargs)
""",
        "ml-wrapper")

    ml = sub(
        ml,
        """    return ModelInterface(
        model_fn=wrapped_model_fn,""",
        """    jit_model.step_fn_no_options = wrapped_model_fn_no_options

    return ModelInterface(
        model_fn=wrapped_model_fn,""",
        "ml-attach")

    try:
        ast.parse(ml)
    except SyntaxError as e:
        fail(f"[model_loader] patched source does not parse: {e}")
    open(ML, "w", encoding="utf-8").write(ml)

# ---------------------------------------------------------------- (b) decode_loop
if not os.path.exists(DL):
    fail(f"{DL} not found")
dl = open(DL, encoding="utf-8").read()

if "_get_decode_core" in dl:
    print("CD-PROPER PATCH: decode_loop already patched")
else:
    dl = sub(
        dl,
        """from tpu_inference.layers.common.attention_metadata import (
    AttentionMetadata, SharedAttentionMetadata)""",
        """from tpu_inference.layers.common.attention_metadata import (
    AttentionMetadata, SharedAttentionMetadata)
from tpu_inference.models.common.compiler_options import \\
    get_step_fn_compiler_options""",
        "dl-import")

    STATIC = """    static_argnames=(
        "model_fn",
        "compute_logits_fn",
        "sample_fn",
        "mesh",
        "static_max_decode_steps",
        "eos_token_id",
        "padding_token_id",
        "dp_size",
        "pad_len",
        "has_experts",
        "expert_shape",
        "expert_dtype",
        "layer_name_to_kvcache_index",
        "is_first_rank",
        "is_last_rank",
        "max_logprobs",
        "logprobs_mode",
        "continue_decode_eos_check_interval",
    ),
    donate_argnames=("kv_caches", ),"""

    old_dec = ("@functools.partial(\n    jax.jit,\n" + STATIC + """
    # Hoisted here from the model's step_fun: JAX forbids compiler_options on
    # a nested jit, so they must live on this top-level loop jit instead.
    compiler_options={
        "xla_tpu_all_gather_collective_matmul_mode": "post_spmd_conservative",
        "xla_tpu_reduce_scatter_collective_matmul_mode":
        "post_spmd_conservative",
        "xla_tpu_use_minor_sharding_for_major_trivial_input": "true"
    },
)
def _decode_core(
""")
    dl = sub(dl, old_dec, "def _decode_core_impl(\n", "dl-decorator")

    factory = ("@functools.lru_cache(maxsize=1)\ndef _get_decode_core():\n"
               "    return functools.partial(\n        jax.jit,\n"
               + STATIC.replace("    static_argnames", "        static_argnames")
               .replace('        "model_fn"', '            "model_fn"')
               + "\n        compiler_options=get_step_fn_compiler_options(),\n"
               "    )(_decode_core_impl)\n\n\n")
    # rebuild the static block with consistent 8-space indent instead of string surgery
    inner = "\n".join(("    " + ln) if ln.strip() else ln
                      for ln in STATIC.split("\n"))
    factory = ("@functools.lru_cache(maxsize=1)\n"
               "def _get_decode_core():\n"
               "    # Built lazily (the \"auto\" SparseCore threshold needs live devices) and\n"
               "    # cached so the jit is constructed once, not per decode step.\n"
               "    return functools.partial(\n"
               "        jax.jit,\n"
               + inner + "\n"
               "        compiler_options=get_step_fn_compiler_options(),\n"
               "    )(_decode_core_impl)\n\n\n")
    dl = sub(dl, "def continue_decode(\n", factory + "def continue_decode(\n",
             "dl-factory")

    dl = sub(dl, "lp_ranks_buffer) = _decode_core(",
             "lp_ranks_buffer) = _get_decode_core()(", "dl-callsite")

    try:
        ast.parse(dl)
    except SyntaxError as e:
        fail(f"[decode_loop] patched source does not parse: {e}")
    open(DL, "w", encoding="utf-8").write(dl)

# ---------------------------------------------------------------- verify
ml2 = open(ML, encoding="utf-8").read()
dl2 = open(DL, encoding="utf-8").read()
problems = []
if "jit_model.step_fn_no_options = wrapped_model_fn_no_options" not in ml2:
    problems.append("step_fn_no_options not attached")
if "run_model_no_options = run_model_jit(run_model_impl)" not in ml2:
    problems.append("options-free twin missing")
if "_get_decode_core()(" not in dl2:
    problems.append("decode_loop call site not rewired")
if "compiler_options=get_step_fn_compiler_options()," not in dl2:
    problems.append("decode_loop still not sourcing options from the function")
if '"xla_tpu_use_minor_sharding_for_major_trivial_input": "true"' in dl2:
    problems.append("decode_loop hardcoded literal still present")

if problems:
    fail("; ".join(problems))
print("CD-PROPER PATCH: APPLIED")
