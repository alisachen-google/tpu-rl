"""Keep the KV cache across weight sync (qwen3 item 6).

WHY. VllmSampler.update_params tears the KV pool down before every weight
reshard (collective_rpc("delete_kv_cache")) and rebuilds it after
(reinitialize_kv_cache) -- ~2 s/step on tianyu's SYNC-T timers. At sync time
the pool is empty (all sequences finished) and the resharded weights are
far smaller than free HBM, so the teardown buys nothing.

WHAT. Gate all delete_kv_cache / reinitialize_kv_cache collective_rpc calls
behind SYNC_KEEPKV (default "1" = skip them; set SYNC_KEEPKV=0 to restore
stock behavior). reset_prefix_cache is kept.

FAIL-OPEN. Prints exactly one of:
    SYNC-KEEPKV PATCH: APPLIED
    SYNC-KEEPKV PATCH: FAILED <reason>
"""
import ast
import os
import re
import sys

VS = "/usr/local/lib/python3.12/site-packages/tunix/generate/vllm_sampler.py"


def fail(msg):
    print(f"SYNC-KEEPKV PATCH: FAILED {msg}")
    sys.exit(0)


if not os.path.exists(VS):
    fail(f"{VS} not found")
src = open(VS, encoding="utf-8").read()
if "SYNC_KEEPKV" in src:
    print("SYNC-KEEPKV PATCH: already patched")
    sys.exit(0)

PAT = re.compile(
    r"^([ \t]*)(self\.(?:llm|_driver\.llm_engine)\.collective_rpc\("
    r"\"(?:delete_kv_cache|reinitialize_kv_cache)\"\)[^\n]*)$",
    re.M)
matches = PAT.findall(src)
if len(matches) != 4:
    fail(f"expected 4 kv-cache rpc lines, found {len(matches)}")

def _gate(m):
    ind, call = m.group(1), m.group(2)
    return (f"{ind}if os.environ.get(\"SYNC_KEEPKV\", \"1\") != \"1\":\n"
            f"{ind}  {call}")

src = PAT.sub(_gate, src)
if "import os" not in src:
    src = src.replace("import ", "import os\nimport ", 1)

try:
    ast.parse(src)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")
open(VS, "w", encoding="utf-8").write(src)
if open(VS, encoding="utf-8").read().count("SYNC_KEEPKV") != 4:
    fail("re-verify failed")
print("SYNC-KEEPKV PATCH: APPLIED (SYNC_KEEPKV="
      f"{os.environ.get('SYNC_KEEPKV', '1(default)')})")
