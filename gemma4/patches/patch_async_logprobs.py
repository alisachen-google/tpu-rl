"""Extract per-token logprobs in the background (ported from tianyu-rl-g4-003, env-gated).
tunix VllmSampler.detokenize runs get_logprobs_from_vllm_output per completion after generation
(~9M (token, dict) pairs/step, sequential ~3 s). Wraps LLMEngine.step (after patch_async_detok)
to extract logprobs of each FINISHED request in the shared thread pool while decode continues.
GATE: active only when env ASYNC_LOGPROBS is 'on' or 'verify8' (default off = original path).
Prints ASYNC-LOGPROBS PATCH: APPLIED|FAILED."""
import sys
E = "/usr/local/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py"
T = "/usr/local/lib/python3.12/site-packages/tunix/generate/vllm_sampler.py"
e = open(E).read(); t = open(T).read()
if "_ASYNC_LOGPROBS" in t:
    print("ASYNC-LOGPROBS PATCH: already applied", flush=True); sys.exit(0)
a = ("        logprobs = utils.get_logprobs_from_vllm_output(\n"
     "            list(single_output.token_ids), single_output.logprobs\n"
     "        )\n")
b = ("        logprobs = _AL_LOOKUP(multi_sampling_output.request_id, idx, single_output)  # _ASYNC_LOGPROBS\n")
if t.count(a) != 1 or "_AD_POOL" not in e:
    print("ASYNC-LOGPROBS PATCH: FAILED anchor tunix=%d engine_pool=%s" % (t.count(a), "_AD_POOL" in e), flush=True); sys.exit(0)
e += r'''
# --- _ASYNC_LOGPROBS -------------------------------------------------------------
import os as _al_os
_AL = {}
def _al_extract_one(ids, lps):
    if not lps or lps[0] is None:
        return []
    if len(lps) != len(ids):
        raise AssertionError("log probs has %d number of items != %d token ids" % (len(lps), len(ids)))
    return [lp[t].logprob for t, lp in zip(ids, lps)]
def _al_extract(o):
    return [_al_extract_one(list(c.token_ids), c.logprobs) for c in o.outputs]
_al_orig_step = LLMEngine.step
def _al_step(self, *a, **k):
    outs = _al_orig_step(self, *a, **k)
    if _al_os.environ.get("ASYNC_LOGPROBS", "off") in ("on", "verify8"):
        for o in outs:
            if getattr(o, "finished", False):
                _AL[o.request_id] = _AD_POOL.submit(_al_extract, o)
    return outs
LLMEngine.step = _al_step
'''
t = t.replace(a, b) + r'''
# --- _ASYNC_LOGPROBS lookup ------------------------------------------------------
import os as _al_os
_AL_MODE = _al_os.environ.get("ASYNC_LOGPROBS", "off")
_AL_STATS = {"hit": 0, "miss": 0, "checked": 0, "mismatch": 0}
def _AL_LOOKUP(request_id, idx, single_output):
    res = None
    if _AL_MODE in ("on", "verify8"):
        try:
            from vllm.v1.engine import llm_engine as _le
            fut = _le._AL.get(request_id)
            if fut is not None:
                res = fut.result()[idx]
        except Exception:
            res = None
    if res is None:
        _AL_STATS["miss"] += 1
        res = utils.get_logprobs_from_vllm_output(list(single_output.token_ids), single_output.logprobs)
    else:
        _AL_STATS["hit"] += 1
        if _AL_MODE == "verify8" and (_AL_STATS["hit"] % 8) == 0:
            ref = utils.get_logprobs_from_vllm_output(list(single_output.token_ids), single_output.logprobs)
            _AL_STATS["checked"] += 1
            if list(ref) != list(res):
                _AL_STATS["mismatch"] += 1
                if _AL_STATS["mismatch"] <= 3:
                    print("ASYNC-LOGPROBS MISMATCH rid=%s idx=%d len %d vs %d" % (request_id, idx, len(ref), len(res)), flush=True)
                res = ref
    if (_AL_STATS["hit"] + _AL_STATS["miss"]) % 2048 == 0:
        print("ASYNC-LOGPROBS hits=%d misses=%d checked=%d mismatch=%d" % (_AL_STATS["hit"], _AL_STATS["miss"], _AL_STATS["checked"], _AL_STATS["mismatch"]), flush=True)
    return res
'''
compile(e, E, "exec"); compile(t, T, "exec"); open(E, "w").write(e); open(T, "w").write(t)
print("ASYNC-LOGPROBS PATCH: APPLIED", flush=True)
