"""Enable per-item math_verify timeouts to cap the reward-phase long tail.

WHY. math_verify_pool.py grades each completion with parse(...,
parsing_timeout=None) and verify(..., timeout_seconds=None) -- per-item
timeouts DISABLED. A single pathological expression (large nested fraction,
unsimplifiable radical, malformed LaTeX) runs sympy unbounded until the GLOBAL
batch timeout (math_verify_timeout=120). One straggler holds up the parallel
pool, so the reward phase spikes (measured g4r11: 191/126/252s on 3 of 20
steps, vs ~10s median) -- the dominant contributor to the step-time MEAN
(162 vs median 125).

WHAT. Set the per-item parse/verify timeouts to VERIFY_ITEM_TIMEOUT seconds
(default 5) instead of None. A pathological item then costs <=5s (marked
wrong, like any grader failure) instead of running to the global cap. Bounds
the reward tail without changing grading of well-formed answers. math_verify
uses signal-based timeouts, valid here because each item runs in its own pool
WORKER PROCESS (main thread of that process). VERIFY_ITEM_TIMEOUT=0 keeps the
original None (disabled) behavior.

FAIL-OPEN. Prints exactly one of:
    VERIFY-TIMEOUT PATCH: APPLIED
    VERIFY-TIMEOUT PATCH: FAILED <reason>
"""
import ast
import os
import sys


def fail(msg):
    print(f"VERIFY-TIMEOUT PATCH: FAILED {msg}")
    sys.exit(0)


try:
    import importlib.util as _u
    spec = _u.find_spec("maxtext")
    if spec is None or not spec.submodule_search_locations:
        fail("cannot locate maxtext package")
    root = list(spec.submodule_search_locations)[0]
except Exception as e:  # noqa: BLE001
    fail(f"maxtext import probe failed: {e}")

MP = os.path.join(root, "trainers", "post_train", "rl", "math_verify_pool.py")
if not os.path.exists(MP):
    fail(f"{MP} not found")
src = open(MP, encoding="utf-8").read()
if "_VT_ITEM" in src:
    print("VERIFY-TIMEOUT PATCH: already patched")
    sys.exit(0)

n_parse = src.count("parsing_timeout=None")
n_verify = src.count("timeout_seconds=None")
if n_parse < 1 or n_verify < 1:
    fail(f"anchors missing (parsing_timeout=None x{n_parse}, timeout_seconds=None x{n_verify})")

src = src.replace("parsing_timeout=None", "parsing_timeout=_VT_ITEM")
src = src.replace("timeout_seconds=None", "timeout_seconds=_VT_ITEM")

# module-level: _VT_ITEM = int(env) or None if 0
inject = (
    "import os as _vt_os\n"
    "_vt_v = _vt_os.environ.get('VERIFY_ITEM_TIMEOUT', '5')\n"
    "try:\n"
    "  _VT_ITEM = int(_vt_v) if int(_vt_v) > 0 else None\n"
    "except Exception:\n"
    "  _VT_ITEM = 5\n"
)
# place after the module docstring / first import block
lines = src.split("\n")
ins_at = 0
for i, ln in enumerate(lines):
    if ln.startswith("import ") or ln.startswith("from "):
        ins_at = i
        break
lines.insert(ins_at, inject)
src = "\n".join(lines)

try:
    ast.parse(src)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")
open(MP, "w", encoding="utf-8").write(src)
v = open(MP, encoding="utf-8").read()
if "_VT_ITEM" not in v or "timeout_seconds=None" in v or "parsing_timeout=None" in v:
    fail("re-verify failed")
print(f"VERIFY-TIMEOUT PATCH: APPLIED (VERIFY_ITEM_TIMEOUT={os.environ.get('VERIFY_ITEM_TIMEOUT','5(default)')})")
