"""Variant A work-stealing (re-prefill migration) wired into the vLLM DP coordinator.

Attacks the straggler: a DP rank that finishes its sequences sits idle in
collective-permute-done while a few ranks grind the long-sequence tail. This patch,
at each schedule() boundary, migrates sequences from the busiest rank to an idle one
by (a) finishing the seq on the donor (frees its KV) and (b) re-adding it on the
acceptor with prompt = original_prompt + tokens_generated_so_far (re-prefill), so the
acceptor rebuilds KV and continues. The migrated seq's completion stays continuous:
the coordinator accumulates its tokens and relabels the acceptor's internal id back to
the original request id before returning to the engine, so the engine only ever sees
one continuous request.

SAFETY. Everything is gated behind env WS_ENABLE (default "0" -> this patch is a pure
no-op: it only installs hooks that early-return). Every hook is wrapped fail-open: any
exception prints a warning and falls back to stock behaviour, never crashing a run.
Correctness of generation is preserved because the acceptor re-prefills the exact
(prompt + generated) prefix and continues sampling from the correct distribution; the
tokens already generated are carried in the accumulator, not re-emitted. REWARD PARITY
MUST STILL BE VERIFIED on any WS_ENABLE=1 run -- treat a reward deviation as failure.

The offline-validated policy + stitch logic this mirrors lives in
scratchpad/variantA/migration.py (unit-tested). This file is the live integration.

Env knobs: WS_ENABLE (0/1), WS_MIN_IMBALANCE (default 2), WS_MAX_MOVES (default 8),
WS_IDLE_ONLY (default 1 -> only migrate to ranks with 0 active, the safest case).

Prints exactly one of:
    WS-A PATCH: APPLIED
    WS-A PATCH: already applied
    WS-A PATCH: FAILED <reason>
"""
import ast
import os
import sys

P = "/usr/local/lib/python3.12/site-packages/tpu_inference/core/sched/dp_scheduler.py"


def fail(msg):
    print(f"WS-A PATCH: FAILED {msg}", flush=True)
    sys.exit(0)  # non-fatal: let the run start so the failure is visible


if not os.path.exists(P):
    fail(f"{P} not found")
src = open(P, encoding="utf-8").read()
if "_WS_A_WORKSTEAL" in src:
    print("WS-A PATCH: already applied", flush=True)
    sys.exit(0)

# ---- anchor 1: __init__ state (after assigned_dp_rank init) ------------------------
A1 = "        self.assigned_dp_rank: Dict[str, int] = {}  # req_id -> dp_rank\n"
if src.count(A1) != 1:
    fail(f"init anchor count={src.count(A1)}")
INIT = A1 + (
    "        # _WS_A_WORKSTEAL state (gated by env WS_ENABLE; default off=no-op)\n"
    "        import os as _ws_os\n"
    "        self._ws_enable = _ws_os.environ.get('WS_ENABLE', '0') == '1'\n"
    "        self._ws_min_imbalance = int(_ws_os.environ.get('WS_MIN_IMBALANCE', '2'))\n"
    "        self._ws_max_moves = int(_ws_os.environ.get('WS_MAX_MOVES', '8'))\n"
    "        self._ws_idle_only = _ws_os.environ.get('WS_IDLE_ONLY', '1') == '1'\n"
    "        self._ws_accum = {}      # orig req_id -> output tokens seen\n"
    "        self._ws_alias = {}      # acceptor worker id -> orig req_id\n"
    "        self._ws_orig_req = {}   # req_id -> Request\n"
    "        self._ws_migrating = set()  # orig req_ids currently mid-migration\n"
    "        self._ws_counter = 0\n"
    "        if self._ws_enable:\n"
    "            print('WS-A: work-stealing ENABLED min_imbalance=%d max_moves=%d idle_only=%s'\n"
    "                  % (self._ws_min_imbalance, self._ws_max_moves, self._ws_idle_only), flush=True)\n"
)
src = src.replace(A1, INIT, 1)

# ---- anchor 2: cache incoming Request in add_request -------------------------------
A2 = ("        if self._batch_prefills:\n"
      "            self._pending_new_requests.append(request)\n"
      "            self._try_flush_pending()\n"
      "            return\n\n"
      "        self._route_and_forward_request(request)\n")
if src.count(A2) != 1:
    fail(f"add_request anchor count={src.count(A2)}")
A2_NEW = ("        try:\n"
          "            self._ws_orig_req[request.request_id] = request  # _WS_A cache\n"
          "        except Exception:\n"
          "            pass\n" + A2)
src = src.replace(A2, A2_NEW, 1)

# ---- anchor 3: migration hook in schedule() ---------------------------------------
A3 = ("        # Run each scheduler independently\n"
      "        for rank in range(self.dp_size):\n"
      "            self._send_command(rank, SchedulerCommand.SCHEDULE, (args, kwargs))\n")
if src.count(A3) != 1:
    fail(f"schedule anchor count={src.count(A3)}")
src = src.replace(A3, "        self._ws_maybe_migrate()  # _WS_A\n" + A3, 1)

# ---- anchor 4: accumulate + relabel + suppress in update_from_output ---------------
# Hook the point where combined outputs/finished are assembled per client.
A4 = ("            combined_output.engine_index = engine_outputs[0].engine_index\n"
      "            combined_output.outputs = outputs\n"
      "            combined_output.finished_requests = finished_requests\n")
if src.count(A4) != 1:
    fail(f"update_from_output anchor count={src.count(A4)}")
A4_NEW = ("            outputs, finished_requests = self._ws_rewrite_outputs(\n"
          "                outputs, finished_requests)  # _WS_A\n" + A4)
src = src.replace(A4, A4_NEW, 1)

# ---- anchor 5: guard _cleanup_finished_requests from removing migrating reqs -------
A5 = '        """Remove finished requests from our DP rank assignment tracking."""\n'
if src.count(A5) != 1:
    fail(f"cleanup anchor count={src.count(A5)}")
A5_NEW = (A5 +
          "        if getattr(self, '_ws_migrating', None):\n"
          "            finished_req_ids = set(finished_req_ids) - self._ws_migrating  # _WS_A\n")
src = src.replace(A5, A5_NEW, 1)

# ---- append the _WS_A helper methods onto the DPScheduler class --------------------
# Find the DPScheduler class body indentation by appending methods right before a
# known later top-level function. We instead append at EOF as module-level funcs bound
# onto the class, which keeps indentation trivial and robust.
HELPERS = '''

# --- _WS_A_WORKSTEAL helper methods, bound onto DPScheduler ------------------------
def _ws_rewrite_outputs(self, outputs, finished_requests):
    """Accumulate per-request output tokens, relabel acceptor alias -> orig id, and
    suppress 'finished' for requests that were migrated (not truly done)."""
    try:
        if not getattr(self, "_ws_enable", False):
            return outputs, finished_requests
        for o in outputs:
            rid = getattr(o, "request_id", None)
            if rid is None:
                continue
            canon = self._ws_alias.get(rid, rid)
            toks = getattr(o, "new_token_ids", None) or []
            if toks:
                self._ws_accum.setdefault(canon, []).extend(list(toks))
            if rid in self._ws_alias:
                # relabel so the engine sees one continuous request
                try:
                    o.request_id = canon
                except Exception:
                    pass
                if getattr(o, "finished", False):
                    # the migrated continuation finished -> the logical request is done
                    self._ws_migrating.discard(canon)
        if finished_requests:
            fr = set()
            for r in finished_requests:
                canon = self._ws_alias.get(r, r)
                if canon in self._ws_migrating:
                    continue  # donor-abort of a migrated req: hide from engine
                fr.add(canon)
            finished_requests = fr
    except Exception as _e:
        print("WS-A rewrite error (fail-open): %r" % (_e,), flush=True)
    return outputs, finished_requests


def _ws_maybe_migrate(self):
    try:
        if not getattr(self, "_ws_enable", False):
            return
        counts = {}
        for rank in range(self.dp_size):
            self._send_command(rank, SchedulerCommand.GET_REQUEST_COUNTS)
        for rank in range(self.dp_size):
            running, waiting = self._get_result(
                rank, SchedulerCommand.GET_REQUEST_COUNTS)
            counts[rank] = int(running)
        if len(counts) < 2:
            return
        active_by_rank = {}
        for req, rk in list(self.assigned_dp_rank.items()):
            if req in self._ws_alias:   # skip acceptor worker ids
                continue
            active_by_rank.setdefault(rk, []).append(req)
        moves = 0
        while moves < self._ws_max_moves:
            donor = max(counts, key=lambda r: counts[r])
            acceptor = min(counts, key=lambda r: counts[r])
            if self._ws_idle_only and counts[acceptor] != 0:
                break
            if counts[donor] - counts[acceptor] < self._ws_min_imbalance:
                break
            cands = [c for c in active_by_rank.get(donor, [])
                     if c in self._ws_orig_req and c not in self._ws_migrating
                     and len(self._ws_accum.get(c, [])) > 0]
            if not cands:
                break
            R = min(cands, key=lambda c: len(self._ws_accum.get(c, [])))
            if not self._ws_migrate_one(R, donor, acceptor):
                break
            counts[donor] -= 1
            counts[acceptor] += 1
            active_by_rank[donor].remove(R)
            active_by_rank.setdefault(acceptor, []).append(R)
            moves += 1
        if moves:
            print("WS-A: migrated %d sequence(s) this step" % moves, flush=True)
    except Exception as _e:
        print("WS-A migrate error (fail-open): %r" % (_e,), flush=True)


def _ws_migrate_one(self, R, donor, acceptor):
    try:
        import copy as _cp
        from vllm.v1.request import Request as _Req
        from vllm.v1.request import RequestStatus as _RS
        orig = self._ws_orig_req.get(R)
        gen = self._ws_accum.get(R, [])
        if orig is None or not gen or orig.prompt_token_ids is None:
            return False
        sp = _cp.copy(orig.sampling_params)
        try:
            sp.max_tokens = max(1, int(orig.max_tokens) - len(gen))
        except Exception:
            pass
        self._ws_counter += 1
        newid = "%s__ws%d" % (R, self._ws_counter)
        rp = _Req(
            request_id=newid,
            prompt_token_ids=list(orig.prompt_token_ids) + list(gen),
            sampling_params=sp,
            pooling_params=None,
            client_index=getattr(orig, "client_index", 0),
            arrival_time=getattr(orig, "arrival_time", None),
            lora_request=getattr(orig, "lora_request", None),
            cache_salt=getattr(orig, "cache_salt", None),
            priority=getattr(orig, "priority", 0),
        )
        # finish R on donor (frees KV); abort status
        self._send_command(donor, SchedulerCommand.FINISH_REQUESTS,
                           ([R], _RS.FINISHED_ABORTED))
        self._get_result(donor, SchedulerCommand.FINISH_REQUESTS)
        # add the re-prefill continuation on the acceptor
        self._send_command(acceptor, SchedulerCommand.ADD_REQUEST, rp)
        self._get_result(acceptor, SchedulerCommand.ADD_REQUEST)
        self._ws_alias[newid] = R
        self._ws_migrating.add(R)
        self.assigned_dp_rank[R] = acceptor
        self.assigned_dp_rank[newid] = acceptor
        return True
    except Exception as _e:
        print("WS-A migrate_one error (fail-open): %r" % (_e,), flush=True)
        return False


DPScheduler._ws_rewrite_outputs = _ws_rewrite_outputs
DPScheduler._ws_maybe_migrate = _ws_maybe_migrate
DPScheduler._ws_migrate_one = _ws_migrate_one
# --- end _WS_A_WORKSTEAL ----------------------------------------------------------
'''
src = src + HELPERS

try:
    ast.parse(src)
except SyntaxError as e:
    fail(f"patched source does not parse: {e}")
open(P, "w", encoding="utf-8").write(src)

v = open(P, encoding="utf-8").read()
problems = []
for needle in ("_ws_maybe_migrate()", "_ws_rewrite_outputs(", "_WS_A cache",
               "DPScheduler._ws_migrate_one = _ws_migrate_one"):
    if needle not in v:
        problems.append(f"missing {needle}")
try:
    ast.parse(v)
except SyntaxError as e:
    problems.append(f"reparse: {e}")
if problems:
    fail("; ".join(problems))
print("WS-A PATCH: APPLIED", flush=True)
