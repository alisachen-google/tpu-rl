#!/usr/bin/env python3
"""Generate launcher recipes (JSON, loaded by `python -m benchmark.tpu.launch_server --config @file`)
for the 1P1D configuration matrix (Section 3.2 of the test plan) and the monolithic golden server.

  Config  prefill                                   decode
  A       radix OFF                                 radix force-off (raiden receiver layout)
  B       radix ON                                  -
  C       B + HiCache (raiden host pool)            -
  D       C + EAGLE3 (gpt-oss only)                 EAGLE3, cpu_tensor retraction
  E       D + decode radix + decode HiCache         (10/30)
  G       monolithic single engine (golden), same model/KV/page/compile flags, no disaggregation

Shapes set context_length / max_total_tokens / max_running_requests: 128k, 256k, 512k, 1m.
Usage: make_recipes.py --out ~/pd_disagg/recipes [--models gpt-oss,qwen3-vl] [--configs A,B,C,D,E,G] [--shapes 128k,1m]
"""
import argparse, json, os

HF = "/mnt/data/tt/hf-cache/hub"
MODELS = {
    "gpt-oss": {
        "model_path": f"{HF}/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
        "served_model_name": "gpt-oss-120b",
        "page_size": 64,
        "mem_fraction_static": 0.45,
        "reasoning_parser": "gpt-oss",
        "draft": f"{HF}/models--lmsys--EAGLE3-gpt-oss-120b-bf16/snapshots/21e624039fdac60defaaa74344e03da5627a8678",
        "env": {"SGLANG_TPU_FUSED_MOE_SC_GATHER": "0"},
    },
    "qwen3-vl": {
        "model_path": f"{HF}/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c",
        "served_model_name": "qwen3-vl-30b",
        "page_size": 128,
        "mem_fraction_static": 0.50,
        "draft": None,
        "env": {},
    },
}
# raiden host staging needs ~9.1 GiB per transfer slot at 1M context (128 GiB default / 8 ranks / 2 engines = 8 GiB -> startup ValueError)
RAIDEN_HOST_POOL_BY_SHAPE = {"128k": "128", "256k": "128", "512k": "256", "1m": "320"}
SHAPES = {  # ISL -> (context_length, max_total_tokens, max_running_requests)
    "128k": (131072 + 9216, 786432, 8),
    "256k": (262144 + 9216, 786432, 4),
    "512k": (524288 + 9216, 1114112, 2),
    "1m":   (1047552 + 9216, 1114112, 1),
}
LIBTPU_INIT_ARGS = " ".join([
    "--xla_tpu_enable_async_collective_fusion=true", "--xla_tpu_overlap_compute_collective_tc=true",
    "--xla_tpu_enable_async_collective_fusion_multiple_steps=true", "--xla_tpu_enable_latency_hiding_scheduler=true",
    "--xla_latency_hiding_scheduler_rerun=10", "--xla_tpu_all_gather_collective_matmul_mode=post_spmd_conservative",
    "--xla_tpu_reduce_scatter_collective_matmul_mode=post_spmd_conservative",
    "--xla_tpu_enable_sparse_core_collective_offload_all_reduce=false", "--xla_tpu_offload_gather_to_sparsecore=false",
    "--xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=false",
])
COMMON_ENV = {
    "SGLANG_PLATFORM": "tpu", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HOME": HF.rsplit("/", 1)[0],
    "TORCH_LOGS": "+recompiles",
    "TPU_ENABLE_DIAGNOSTIC_COORDINATOR": "0", "TPU_DISABLE_DIAGNOSTIC_COORDINATOR": "1",
    "TPU_DIAGNOSTIC_HARVESTER_INTERVAL_SEC": "86400", "TPU_ENABLE_TELEMETRY_HARVESTING": "0", "TPU_ENABLE_TELEMETRY": "0",
    "ENABLE_TPU_TELEMETRY": "0", "TPU_TELEMETRY_INTERVAL_SECONDS": "86400", "TPU_TELEMETRY_INTERVAL_SEC": "86400",
    "SGLANG_TPU_PRELOAD_TELEMETRY_STUB": "1", "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "0",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1", "SGLANG_TPU_OVERLAP_PROFILE": "1",
    "TORCH_TPU_INTERNAL_MATERIALIZE_COLLECTIVE_TENSORS": "false", "LIBTPU_INIT_ARGS": LIBTPU_INIT_ARGS,
    # disaggregation / raiden
    "TPU_RAIDEN_CONTROL_PLANE_BACKEND": "grpc", "SGLANG_RAIDEN_KV_PORT": "31000",
    "SGLANG_RAIDEN_TIMEOUT_S": "600", "SGLANG_RAIDEN_HOST_POOL_GB": "128",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "7200", "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "7200",
    "LB_PORT": "8080", "HEALTH_TIMEOUT": "5400",
}


def base_args(m, shape, kv_dtype):
    ctx, pool, mr = SHAPES[shape]
    bs = [b for b in (1, 2, 4, 8, 16) if b < mr] + [mr]
    a = {
        "device": "tpu", "dtype": "bfloat16", "host": "0.0.0.0", "port": 30000, "grammar_backend": "none",
        "model_path": m["model_path"], "served_model_name": m["served_model_name"],
        "tp_size": 8, "ep_size": 8, "attention_backend": "pallas_tpu_rpa_v3",
        "page_size": m["page_size"], "kv_cache_dtype": kv_dtype, "quantization": None,
        "context_length": ctx, "max_total_tokens": pool, "max_running_requests": mr,
        "chunked_prefill_size": 8192, "max_prefill_tokens": 8192,
        "tpu_compile_tokens": [512, 1024, 2048, 4096, 8192], "tpu_compile_bs": bs,
        "enable_torch_compile": True, "torch_compile_max_bs": mr,
        "mem_fraction_static": m["mem_fraction_static"], "sampling_backend": "pytorch",
        "skip_server_warmup": True, "enable_metrics": True, "enable_cache_report": True,
        "watchdog_timeout": 3600,
    }
    if m.get("reasoning_parser"):
        a["reasoning_parser"] = m["reasoning_parser"]
    return a


HICACHE = {"enable_hierarchical_cache": True, "hicache_ratio": 2, "hicache_io_backend": "raiden",
           "hicache_mem_layout": "page_first_fused", "hicache_write_policy": "write_through"}


def spec_args(m):
    return {"speculative_algorithm": "EAGLE3", "speculative_draft_model_path": m["draft"],
            "speculative_num_steps": 3, "speculative_eagle_topk": 1, "speculative_num_draft_tokens": 4}


def pd_recipe(model, cfg, shape):
    m = MODELS[model]
    spec = cfg in ("D", "E")
    if spec and not m["draft"]:
        return None
    kv = "bfloat16" if spec else "fp8_e4m3"   # spec recipes use bf16 KV (benchmark/tpu/recipe/configs/gpt_oss_120b/pd_singlehost_spec.py)
    p = base_args(m, shape, kv); d = base_args(m, shape, kv)
    p.update(disaggregation_mode="prefill", disaggregation_transfer_backend="raiden", disaggregation_bootstrap_port=8998)
    d.update(disaggregation_mode="decode", disaggregation_transfer_backend="raiden",
             hicache_io_backend="raiden", hicache_mem_layout="page_first_fused")  # raiden receiver host layout
    p_env, d_env = dict(m["env"]), dict(m["env"])
    if cfg == "A":
        p["disable_radix_cache"] = True
    else:
        p["disable_radix_cache"] = False
    if cfg in ("C", "D", "E"):
        p.update(HICACHE)
    if spec:
        p.update(spec_args(m)); d.update(spec_args(m))
        d["disaggregation_decode_retraction_backup"] = "cpu_tensor"
    if cfg == "E":
        d.update(HICACHE)
        d.update(disaggregation_decode_enable_radix_cache=True, disaggregation_decode_retraction_backup="cpu_tensor")
        d_env["SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND"] = "python"   # required by tpu/pd_decode_swa_hicache.py
    env = dict(COMMON_ENV); env["SGLANG_RAIDEN_HOST_POOL_GB"] = RAIDEN_HOST_POOL_BY_SHAPE[shape]
    return {"group": {"hosts": 2, "prefill_engines": 1, "decode_engines": 1},
            "runtime_envs": env,
            "prefill": {"server_args": p, "runtime_envs": p_env},
            "decode": {"server_args": d, "runtime_envs": d_env}}


def golden_recipe(model, shape, kv):
    m = MODELS[model]
    a = base_args(m, shape, kv)
    a["disable_radix_cache"] = True
    # goldens: allow a few concurrent requests so short-ISL prompt sets finish quickly
    a.update(max_running_requests=4, torch_compile_max_bs=4, tpu_compile_bs=[1, 2, 4])
    env = {**COMMON_ENV, **m["env"]}
    for k in ("LB_PORT",):
        env.pop(k, None)
    return {"server_args": a, "runtime_envs": env}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.expanduser("~/pd_disagg/recipes"))
    ap.add_argument("--models", default="gpt-oss,qwen3-vl")
    ap.add_argument("--configs", default="A,B,C,D,E,G")
    ap.add_argument("--shapes", default="128k,256k,512k,1m")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = 0
    for model in a.models.split(","):
        for shape in a.shapes.split(","):
            for cfg in a.configs.split(","):
                if cfg == "G":
                    for kv in ("fp8_e4m3", "bfloat16"):
                        r = golden_recipe(model, shape, kv)
                        path = os.path.join(a.out, f"{model}_G_{shape}_{kv}.json")
                        json.dump(r, open(path, "w"), indent=1); n += 1
                    continue
                r = pd_recipe(model, cfg, shape)
                if r is None:
                    continue
                path = os.path.join(a.out, f"{model}_{cfg}_{shape}.json")
                json.dump(r, open(path, "w"), indent=1); n += 1
    print(f"wrote {n} recipes to {a.out}")


if __name__ == "__main__":
    main()
