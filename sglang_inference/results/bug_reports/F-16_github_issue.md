# GitHub issue draft for sgl-project/sglang-torchtpu

**Title:** [PD] `--max-queued-requests` and priority-based queue eviction are silently ignored in disaggregated mode (`_abort_on_queued_limit` runs only when `disaggregation_mode == NULL`)

**Labels:** bug, pd, scheduling

---

## Summary

`--max-queued-requests N` bounds the scheduler's waiting queue: when the queue is full a newcomer gets HTTP 503 "The request queue is full",
and with `--enable-priority-scheduling` the lowest-priority waiting request is evicted instead if the newcomer outranks it ("The request is
aborted by a higher priority request"). In a prefill or decode engine neither happens: `_add_request_to_queue` calls `_abort_on_queued_limit`
only inside `if self.disaggregation_mode == DisaggregationMode.NULL`. The flag is accepted and the queue grows without bound; the
priority eviction path can never run.

**Expected behavior.** Either the limit applies to the prefill engine's waiting queue (the natural place: it is where requests wait for
chunked prefill, and the decode's prealloc queue is already bounded by KV space), or the server refuses the flag in PD mode with a clear
message. Silently ignoring an operator's back-pressure knob is the worst of the three, because a load balancer or client that relies on 503
to shed load never sees one.

**Impact.** Medium for production PD: no back-pressure, no priority-based shedding, queue latency unbounded under overload; issue #1161 asks
to validate priority scheduling with PD, and this path is part of it. S2 / P2.

## Environment

- `sglang-torchtpu` main @ `2cec1cf194` (also at `09d1244c26`), 1 prefill + 1 decode, TPU v7x, gpt-oss-120b, Config C (radix on, HiCache)
  plus `--enable-priority-scheduling --disable-priority-preemption --max-queued-requests 4` on both engines.

## Code trace

`_add_request_to_queue`
([scheduler.py#L3448-L3462](https://github.com/sgl-project/sglang-torchtpu/blob/2cec1cf194/python/sglang/srt/managers/scheduler.py#L3448-L3462)):

```python
        if self.disaggregation_mode == DisaggregationMode.NULL:
            if self._abort_on_queued_limit(req):
                self._release_aborted_request(req)
                return
            self._prefetch_kvcache(req)
```

`_abort_on_queued_limit`
([scheduler.py#L3517-L3560](https://github.com/sgl-project/sglang-torchtpu/blob/2cec1cf194/python/sglang/srt/managers/scheduler.py#L3517-L3560))
holds the whole limit and priority-eviction logic and has no other caller (`grep -rn _abort_on_queued_limit python/sglang/srt`). The gate was
introduced with "Fix metrics and request tracing (TimeStats)" (2d62af6be5). `max_queued_requests` is otherwise only read for metrics.

## How we detected it

Issue #1161 battery, queue-limit configuration (`gpt-oss_C_128k_ps_q.json`): 8 slots held by priority-5 decodes, 4 priority-5 fillers
queued, then newcomers with priority 3 (expect 503), 5 (expect 503) and 9 (expect one filler evicted). All three newcomers returned 200 and
nothing was evicted; both engines' logs contain no "queue is full" line. The same probe on a monolithic server with the same flags
(`gpt-oss_G_128k_ps_q.json`: 4 slots, `--max-queued-requests 4`, priority scheduling, no preemption) behaves as documented: newcomer priority 3
→ **503 "The request queue is full."**, newcomer priority 5 (equal) → 503, newcomer priority 9 → the newest priority-5 waiting request is evicted
with **"The request is aborted by a higher priority request."** and the newcomer is served (`results/1161/mono_ps_q.txt`). The gap is therefore
PD-specific.

## How to reproduce

```bash
# prefill + decode launched with: --enable-priority-scheduling --disable-priority-preemption --max-queued-requests 4 --max-running-requests 8
python scripts/priority_probe.py --url http://<mini-lb>:8080 --decode http://<decode>:30000 --mode q --slots 8
# D1-low-rejected FAIL ... HTTP 200 ; D3-equal-rejected FAIL ... HTTP 200 ; D2-evict-lowest FAIL evicted []
```

## Fix proposal

Apply the limit on the prefill engine as well (`disaggregation_mode in (NULL, PREFILL)`), keeping the decode exempt because its arrivals are
already admitted by the prefill. If the PD semantics are intentionally different, reject `--max-queued-requests` in PD mode at argument
validation with a message that names the alternative (`--max-running-requests` plus the load balancer's own limit). Add a unit test that drives
`_add_request_to_queue` in PREFILL mode with a full queue and asserts the 503 and the priority eviction.

## Related

- #1161 (validate priority scheduling with PD): this is one of its boundary cases.
- No open issue mentions `max-queued-requests` or queue-full behaviour in PD (searched 2026-10-09).

## Evidence

In `alisachen-google/tpu-rl`, directory `sglang_inference/`: `results/1161/ps_q.txt`, `results/phase_1161.log`, recipe
`recipes/gpt-oss_C_128k_ps_q.json`, `results/TEST_PLAN_1161_priority_scheduling.md` §3-D.

---

# Fix validation (PR #1191, 2026-10-10)

Filed as #1188; fix PR https://github.com/sgl-project/sglang-torchtpu/pull/1191 (decode-side queue limit and eviction, backlog counted net of free running slots, empty-queue guard).

## Live validation matrix (TPU v7x, 1 prefill + 1 decode, gpt-oss-120b, raiden, mini-LB)

Head 2f67d8b701 on both hosts for the matrix; the limit-0 fix (bb5bbe6c1b) re-validated afterwards. Config C = radix + HiCache on the prefill, 8 running slots. Probe: `sglang_inference/scripts/queue_limit_probe.py` in alisachen-google/tpu-rl; outputs in `results/fix_validate_1191_boundary/`.

### Limit 4, priority scheduling on (high values first) — Config C

| Case | Result |
|---|---|
| Q1 burst of 8 requests into an idle decode | all 200, none rejected |
| Q2 burst of 12 (8 run + 4 wait = the limit) | all 200, none rejected |
| Q3 equal-priority newcomer into the full queue | **503** "The request queue is full." |
| Q4 lower-priority newcomer | **503** |
| Q5 two priority-9 newcomers in a row | the two **newest** priority-5 waiters evicted (503 "aborted by a higher priority request"), both newcomers 200 |
| Q6 KV after eviction | small fillers are below one page, see Q9 for the visible case |
| Q7 a queued request's client aborts | 5th waiter 503; a priority-9 request that was admitted by eviction then abandoned by its client: its place frees after ~52 s (the abort reaches the decode through the mini-LB; 3 s on a single engine) and the next equal newcomer is admitted (200) |
| Q8 streaming fillers | the evicted one ends with 503, the others stream to completion, newcomer 200 |
| Q9 8k-token fillers | eviction frees the waiter's KV (decode used tokens 60,928 → 0), newcomer 200 |
| Q10 soak, 6 fill/evict cycles with the queue kept full | queue kept full; six newcomers of rising priority (9…14) evict exactly one waiter each, in the expected order (the newest priority-5 fillers first, then the oldest newcomers 9 and 10); the four surviving newcomers are served after the holders finish |
| end of every group | decode `num_used_tokens` 0, 0 scheduler exceptions, both engines healthy; exactly as many 503s on the decode as the cases expect (8) |

### Other configurations

| Configuration | Cases | Result |
|---|---|---|
| limit 4, **low values first** | priority 9 newcomer; priority 1 newcomer | 503; evicts the newest priority-5 waiter and is served |
| **limit 0** | burst into idle; newcomer with slots full; priority 9 with an empty queue | all 200; newcomer with slots full 503; **priority 9 into an empty capped queue 503** (before the limit-0 fix in this PR the eviction branch called `max()` on the empty queue and every decode rank died; found by this matrix) |
| **limit 1** | second equal waiter; priority 9 newcomer | 503; evicts the single waiter |
| limit 4, **priority scheduling off** | 5th waiter; all fillers | 503; all served, no eviction |
| **no limit** (default path) | 13 requests at once | all 200, 5 waited at peak, nothing rejected |
| **KV pressure**: decode pool 262k, 8 × 131k at c=8 | long holders, fillers, newcomer | passes on this head (8/8 long holders, fillers and newcomer all 200, KV back to 0) and on main. One of three runs of this pattern on the PR head stalled for 30 min with zero exceptions: the TPU KV free fence kept the whole pool held behind the last forward while six requests waited for room (log: "KV free fence … pages held 4166"), which is before anything this PR touches; recorded as an intermittent upstream observation |
| **monolithic reference** (4 slots, limit 4, same flags) | Q3/Q4/Q5 | same three outcomes as the pair. Q2 differs: the single-engine intake rejected 4 of 8 burst requests into an idle server, the over-count this PR avoids on the decode by counting free slots (noted as an upstream observation) |

Zero scheduler exceptions in every configuration except the limit-0 crash this matrix found and this PR now fixes.


Probe: `scripts/queue_limit_probe.py` (modes base/low/q0/q1/nops/nolimit/retract/soak); recipes `recipes/gpt-oss_C_128k_ps_q*.json`, `gpt-oss_C_128k_q_nops.json`, `gpt-oss_A_128k_smallpool_ps_q.json`, `gpt-oss_G_128k_ps_q.json`; outputs `results/fix_validate_1191_boundary/`; phase logs `results/phase_1191_*.log`.

### Fix re-validation after redesign as a TPU patch (2026-10-11, PR #1191 head 21859c77df)
sglang_tpu/pd_decode_queue_limit.py (wraps DecodeTransferQueue.pop_transferred and Scheduler._abort_on_queued_limit). base 16 PASS / 1 known probe-design FAIL (Q10), low 3 PASS, q0 4 PASS, nops 3 PASS; prefill 0 exceptions, decode only the 4 launcher tracebacks. Raw: results/fix_validate_1191_patch/, log results/fix_validate_1191_patch.log.

### Final shape of PR #1191: option 1 (validation-only), 2026-10-12, head 8f578ae4ae
Per review, the decode-side admission was dropped. TpuSRTPlatform.apply_server_args_defaults now refuses --max-queued-requests under a PD mode, and refuses limit 0 + --enable-priority-scheduling. Live: PD decode launch with the flag exits with the message (no processes); monolithic limit 0 + priority launch exits with the message; main with the same monolithic recipe: first burst -> connection errors, 'max() iterable argument is empty' in all 8 ranks, /health dead, 0 schedulers (results/fix_validate_1191_option1/, logs fix_validate_1191_option1{,b}.log). Note: upstream counts the newcomer itself, so limit 0 refuses every request even into an idle server. The decode-side admission moved to the door (option 2) in PR #1299.

### Option 2: door admission (PR #1299, 2026-10-12, head b9da83f34a)
Admission decided at the decode's door (DecodePreallocQueue._init_receiver) before any prefill work; ABORT + reason to the prefill ranks, prefill twin answers 503, router relays 503. First design (abort right after receiver construction) hung the client: receiver only learns prefill ranks in init(); router waits for the prefill's status. Live: 6 recipes, 63 checks, 63 pass (results/fix_validate_door2, door4, door5; probe scripts/door_probe.py). No prefill batch for refused requests; 1-2 non-zero-rank "Prefill bootstrap failed" warnings per run from the all-reduced poll racing the ABORT frame (cosmetic).
