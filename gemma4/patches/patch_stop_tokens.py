"""GEMMA4: tunix VllmSampler clobbers vLLM's stop set to [tokenizer.eos_id()] -- for
google/gemma-4-E2B-it that is <eos>=1 only, while the chat model ends turns with ids 106/50
(generation_config eos_token_id=[1,106,50]) -> most completions run to the 8192 cap
(g4r19 measured truncation 64->78% RISING across steps; tianyu g4-001c: 99.4% capped).
Union in EXTRA_STOP_TOKEN_IDS from env. Ported from tianyu-rl-g4-003.
Prints STOP-TOKENS PATCH: APPLIED|FAILED and, at runtime, STOP-TOKENS ids=[...] once."""
import sys
T = "/usr/local/lib/python3.12/site-packages/tunix/generate/vllm_sampler.py"
t = open(T).read()
if "_EXTRA_STOP" in t:
    print("STOP-TOKENS PATCH: already applied", flush=True); sys.exit(0)
a = "      sampling_params.stop_token_ids = [self.tokenizer.eos_id()]\n"
if t.count(a) != 1:
    print("STOP-TOKENS PATCH: FAILED anchor %d" % t.count(a), flush=True); sys.exit(0)
b = ("      sampling_params.stop_token_ids = _extra_stop_ids(self.tokenizer.eos_id())  # _EXTRA_STOP\n")
t = t.replace(a, b) + r'''
# --- _EXTRA_STOP -----------------------------------------------------------------
import os as _es_os, json as _es_json
_ES_PRINTED = [False]
def _extra_stop_ids(eos_id):
    ids = [eos_id]
    try:
        ids += [int(x) for x in _es_json.loads(_es_os.environ.get("EXTRA_STOP_TOKEN_IDS", "[]"))]
    except Exception:
        pass
    out = sorted(set(int(i) for i in ids if i is not None))
    if not _ES_PRINTED[0]:
        print("STOP-TOKENS ids=%s" % out, flush=True); _ES_PRINTED[0] = True
    return out
'''
compile(t, T, "exec"); open(T, "w").write(t); print("STOP-TOKENS PATCH: APPLIED", flush=True)
