## Live validation (TPU v7x, 1 prefill + 1 decode, gpt-oss-120b, raiden, mini-LB)

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
| Q7 a queued request's client aborts | __Q7__ |
| Q8 streaming fillers | the evicted one ends with 503, the others stream to completion, newcomer 200 |
| Q9 8k-token fillers | eviction frees the waiter's KV (decode used tokens 60,928 → 0), newcomer 200 |
| Q10 soak, 6 fill/evict cycles with the queue kept full | __Q10__ |
| end of every group | decode `num_used_tokens` 0, 0 scheduler exceptions, both engines healthy; exactly as many 503s on the decode as the cases expect (8) |

### Other configurations

| Configuration | Cases | Result |
|---|---|---|
| limit 4, **low values first** | priority 9 newcomer; priority 1 newcomer | 503; evicts the newest priority-5 waiter and is served |
| **limit 0** | burst into idle; newcomer with slots full; priority 9 with an empty queue | all 200; __Q0__ |
| **limit 1** | second equal waiter; priority 9 newcomer | 503; evicts the single waiter |
| limit 4, **priority scheduling off** | 5th waiter; all fillers | 503; all served, no eviction |
| **no limit** (default path) | 13 requests at once | all 200, 5 waited at peak, nothing rejected |
| **KV pressure**: decode pool 262k, 8 × 131k at c=8 | long holders, fillers, newcomer | __RETRACT__ |
| **monolithic reference** (4 slots, limit 4, same flags) | Q3/Q4/Q5 | same three outcomes as the pair. Q2 differs: the single-engine intake rejected 4 of 8 burst requests into an idle server, the over-count this PR avoids on the decode by counting free slots (noted as an upstream observation) |

Zero scheduler exceptions in every configuration except the limit-0 crash this matrix found and this PR now fixes.
