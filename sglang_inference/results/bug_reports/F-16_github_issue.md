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
