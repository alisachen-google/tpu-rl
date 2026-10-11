
### Re-validation of the TPU-patch design (head 21859c77df, 2026-10-11)

Same hosts, recipes and probe as the table above (gpt-oss-120b PD, prefill a / decode b, mini-LB). The patched head produced the same verdicts as the shared-code version on every case.

| Recipe | Checks | Result |
|---|---|---|
| base (`--max-queued-requests 4`, priority on) | Q1–Q9: idle burst admitted, limit edge, equal/lower newcomer 503, eviction cascade (newest fillers first), client-abort frees a place, streaming and 8k-prompt fillers evicted, decode idle after each block | 16 PASS, 1 FAIL (Q10, the probe-design case documented above: fillers finish before the next cycle; unchanged from the earlier head) |
| low (`--schedule-low-priority-values-first`) | worse newcomer 503; better newcomer admitted, newest filler evicted; idle after | 3 PASS |
| q0 (`--max-queued-requests 0`, priority on) | burst into idle admitted; any waiter 503; empty queue with priority 9 → 503 instead of the crash that ended all ranks on main; idle after | 4 PASS |
| nops (limit 4, priority off) | 5th waiter 503; fillers all served, nothing evicted; idle after | 3 PASS |

Scheduler exceptions: prefill 0 in every run; decode shows only the launcher's 4 connection-refused tracebacks from the early `freeze_gc` call, present in every run. The 503s on the decode side are themselves the proof the patch is installed: main never rejects there.
