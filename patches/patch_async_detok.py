"""Overlap detokenization with the rollout tail (ported from tianyu-rl-g4-003).
Wrap vllm LLMEngine.step: each finished RequestOutput is decoded in a thread pool while decode
continues; tunix VllmSampler.detokenize uses the precomputed text (falls back to synchronous
decode; prints ASYNC-DETOK hits/misses). NOTE: supersedes patch_fast_detok's batch decode for
finished-in-window requests; both coexist (fast_detok covers misses).
Prints ASYNC-DETOK PATCH: APPLIED | FAILED."""
import sys
E = "/usr/local/lib/python3.12/site-packages/vllm/v1/engine/llm_engine.py"
T = "/usr/local/lib/python3.12/site-packages/tunix/generate/vllm_sampler.py"
e = open(E).read(); t = open(T).read()
if "_ASYNC_DETOK" in e:
    print("ASYNC-DETOK PATCH: already applied", flush=True); sys.exit(0)
a = ("        decoded_outputs[idx].append(\n"
     "            self.tokenizer.decode(single_output.token_ids)\n"
     "        )\n")
b = ("        _ad_pre = _AD_LOOKUP(multi_sampling_output.request_id, idx)  # _ASYNC_DETOK\n"
     "        if _ad_pre is None:\n"
     "          decoded_outputs[idx].append(self.tokenizer.decode(single_output.token_ids))\n"
     "        else:\n"
     "          decoded_outputs[idx].append(_ad_pre)\n")
if t.count(a) != 1:
    print("ASYNC-DETOK PATCH: FAILED tunix anchor %d (fast_detok applied first? order matters: async_detok BEFORE fast_detok)" % t.count(a), flush=True); sys.exit(0)
e = e + "\n\n# --- _ASYNC_DETOK: decode finished requests in the background --------------------\nimport concurrent.futures as _ad_cf\n_AD_POOL = _ad_cf.ThreadPoolExecutor(max_workers=4)\n_AD_TEXT = {}\n_ad_orig_step = LLMEngine.step\ndef _ad_decode(tok, ids):\n    if hasattr(tok, 'batch_decode'):\n        return tok.batch_decode(ids)\n    return [tok.decode(i) for i in ids]\ndef _ad_step(self, *a, **k):\n    outs = _ad_orig_step(self, *a, **k)\n    import os as _ad_os\n    tok = getattr(self, 'tokenizer', None)\n    if tok is not None and _ad_os.environ.get('ASYNC_DETOK','0')=='1':\n        for o in outs:\n            if getattr(o, 'finished', False):\n                ids = [list(c.token_ids) for c in o.outputs]\n                _AD_TEXT[o.request_id] = _AD_POOL.submit(_ad_decode, tok, ids)\n    return outs\nLLMEngine.step = _ad_step\n"
t = t.replace(a, b) + "\n\n# --- _ASYNC_DETOK lookup ---------------------------------------------------------\n_AD_STATS = {'hit': 0, 'miss': 0}\ndef _AD_LOOKUP(request_id, idx):\n    try:\n        from vllm.v1.engine import llm_engine as _le\n        fut = _le._AD_TEXT.get(request_id)\n        if fut is None:\n            _AD_STATS['miss'] += 1; return None\n        texts = fut.result(); _AD_STATS['hit'] += 1\n        if (_AD_STATS['hit'] + _AD_STATS['miss']) % 2048 == 0:\n            print('ASYNC-DETOK hits=%d misses=%d' % (_AD_STATS['hit'], _AD_STATS['miss']), flush=True)\n        return texts[idx]\n    except Exception:\n        _AD_STATS['miss'] += 1; return None\n"
compile(e, E, "exec"); compile(t, T, "exec"); open(E, "w").write(e); open(T, "w").write(t)
print("ASYNC-DETOK PATCH: APPLIED", flush=True)
