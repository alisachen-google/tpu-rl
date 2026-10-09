# PD Disaggregation Test Report (pointer and status addendum)

The full report, with the overall plan, the categorised issue list (ours and the open upstream issues) and the per-case results, is the live
document **PD Disaggregation Test Report**: https://claude.ai/artifact/E5e5mj5ydXPVUehp2KrGe7 (export to Markdown/PDF from the page).
It was written on 2026-10-09 at about 04:30 UTC from `TEST_CASES.md`, `FINDINGS.md`, `bug_reports/PRIORITIES.md`, `bug_reports/INDEX.md`,
`bug_reports/OVERLAP.md` and `TEST_PLAN_1161_priority_scheduling.md` in this directory, which remain the primary records.

## Status addendum (07:50 UTC, after the document was written)

| Item | Outcome |
|---|---|
| PR #1181 (F-11 fix, issue #1142) | **Ready for review**, all 28 CI checks green (CPU shards, build, TPU unit tests, stage-B e2e); scenario battery TC-17 PASS (A, small decode pool, 1M four slots, B/C regression, Qwen text); monolithic before/after: main stalls 1,800 s, fix answers in 25 s |
| Issue #1145 (F-8) | fixed upstream (bab49bcec2 + follow-ups), verified with our six-abort reproduction on main 2cec1cf194; closing comment drafted (`prs/COMMENT_1145_fixed_upstream.md`) |
| Issue #1161 (priority scheduling) | first and second battery plus E4 complete (TC-18): ordering in both directions and FIFO ties PASS; no starvation; no preemption of running decodes in PD by construction (O-6); prefill head-of-line blocking by a long low-priority prompt (O-7); prefix reuse, HiCache host reload and image-prefix reuse across priorities with byte-identical output; `priority` metrics label; 20-abort storm clean; 15-min soak 5,035 requests, 0 errors, TTFT ordered by priority; monolithic preemption, preemptor/victim aborts and repeated preemption clean; **F-16**: `--max-queued-requests` and priority eviction inert in PD (monolithic control passes) |
| New drafts | F-15 (mini-LB 500 on empty `input_ids`), F-16 (queue limit inert in PD) in `bug_reports/`; not yet filed |
| F-10 | confirmed still present on upstream main 2cec1cf194 (1920×1080 image kills the Qwen3-VL prefill) |
| Monitoring | Prometheus + Grafana on the decode host (`monitoring/`), 37-panel PD dashboard, started with every pair launch |
