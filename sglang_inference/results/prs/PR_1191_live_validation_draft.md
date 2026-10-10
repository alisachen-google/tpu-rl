## Live validation (TPU v7x, 1 prefill + 1 decode, gpt-oss-120b, raiden, mini-LB)

Head 2f67d8b701 on both hosts for the matrix; the limit-0 fix (bb5bbe6c1b) re-validated afterwards. Config C = radix + HiCache on the prefill, 8 running slots.

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
