## Live validation

gpt-oss-120b PD pair (prefill host a, decode host b, PD router on a), raiden transfer, head b9da83f34a (stacked on #1191 d4fe0e7376). Probe: `door_probe.py`; each recipe starts a fresh pair.

| Recipe | Checks | Result |
|---|---|---|
| limit 4, priority on (`C_128k_ps_q`) | 20 into idle → 12 admitted (8 slots + 4 waiters), 8 refused, no errors; full queue → 503 in 20 ms with `The request queue is full.` through the router; higher-priority newcomer 503 (no eviction in this version); 3 refused 8k prompts → 503 in 20–30 ms, **no prefill batch ≥ 4000 tokens in the prefill log**; streaming newcomer 503; 16-request storm with 1.5 s client timeouts → mix of 503/200, engines idle after, next request 200; 5 soak cycles each (3 refused, 12 served, idle, prefill bootstrap queue 0); 3 greedy prompts identical idle vs under pressure; decode and prefill idle and clean at every checkpoint | 15 PASS |
| limit 4, priority off (`C_128k_q_nops`) | same checks minus the priority one | 14 PASS |
| limit 4, low values first (`C_128k_ps_q_low`) | same checks | 15 PASS |
| limit 0, priority on (`C_128k_ps_q0`) | 12 into idle → 8 admitted (the slots), 4 refused; waiter #1 → 503 in 40 ms; priority-9 newcomer with nothing to evict → 503, scheduler alive | 7 PASS |
| limit 1 (`C_128k_ps_q1`) | 12 into idle → 9 admitted, 3 refused; waiter #2 → 503 in 20 ms; priority-9 newcomer → 503 | 7 PASS |
| small decode pool, 262k tokens (`A_128k_smallpool_ps_q`) | 7 holders with 32k-token prompts generating 16k tokens each outgrow the pool: a retraction is observed after 98 s (pool at 256k used, 6 running); 4 newcomers sent while a request is retracted are all answered (first waits 42 s for pages, then 200; no refusal because retracted + waiting never exceeded the limit of 4 with free slots counted); all 7 holders finish (no deadlock after retraction); decode and prefill idle and clean | 5 PASS |

Total: 63 checks, 63 pass. A first attempt at the retraction recipe with 8 × 20k-token holders never retracted (the pool held them), so it was rerun with the load above.

Prefill side, per recipe: one `decode refused room … rid` info line per rank per refusal (e.g. 287 lines for 28 refused rids in the low-first run), no large prefill batches, scheduler exceptions 0 on both engines. One or two refusals per run reach a non-zero rank's `handle_bootstrap_failure` through the all-reduced poll before that rank has processed the ABORT frame; those ranks log the generic "Prefill bootstrap failed" warning. Rank 0, which answers the client, always took the 503 path (every refused request above received the 503). Cosmetic; noted for a follow-up.

Earlier head (first design, `refuse()` right after constructing the receiver): the refused request's client got a connection error and the prefill logged no failure, because the receiver only learns the prefill ranks in `init()`; that is what led to the current design.
