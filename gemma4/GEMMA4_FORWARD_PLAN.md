# Forward Optimization Plan: g4r19 → optimal, step by step

Built from the exact config diff between g4r19 (our launcher) and g4r32b (champion).
Each step changes ONE thing (or one named bundle) and is measured with an 8-10 step run.

## The config diff (g4r19 → g4r32b)

| Dimension | g4r19 | g4r32b |
|---|---|---|
| Prompt | canonical `<|turn>` template | gemma-3-style textual (`gemma_data.py`) + stop-ids [1,106,50] |
| CD window | 64 | 512 |
| Textual `<end_of_turn>` stop | off | ON (our patch) |
| Async host work | none (serial rewards/detok/logprobs) | async reward+detok+logprobs |
| XLA proxy flags | none | 29-flag ladder (stability-required ≥ cd512) |
| Trainer | TMBS 4, HBM 0.25 | TMBS 8, HBM 0.3 |
| Engine shape | mnbt 8448, block auto, prefix on, best-rank routing | mnbt 33280, block 128 pinned, prefix OFF, rr routing, eos-interval 8 |
| Patch stack | our roster (26) | tianyu 34-patch stack + our stop/dump/vtmo/partial |
| Reward spike guard | vtmo only | vtmo + _AR_PARTIAL |

## The ladder (F-steps)

| Step | Run | Delta vs previous | Status / expected |
|---|---|---|---|
| F0 | **g4r19b** (RUNNING) | baseline re-run (stop OFF, fixed reward weights, dumps on) | expect ~175 median |
| F1 | g4r19c | + textual prompt + stop-ids (env-only: DL_DSPROC, DL_STOPIDS) | prompt term on our launcher; expect osl/quality shift, modest perf |
| F2 | g4r34 | **platform swap**: reference-lineage manifest @ cd64, textual stop OFF | the base-bundle hop (XLA ladder + TMBS8/HBM0.3 + engine shape + their patch variants + async); internals partially priced (async −5.6 via g4r35) |
| F3 | g4r33 ✅ | + textual stop @ cd64 | **51.7 measured** |
| F4 | g4r32b ✅ | + window 512 | **40.7 measured** |
| F5 | **g4r32c** (RUNNING, priority) | + reward-spike fix (vtmo + _AR_PARTIAL) | median ~40.7, MEAN 60.9→~44; the production config |

F2 is the one aggregate hop: it cannot decompose on our launcher (cd512 instability without
the full reference stack — g4r21c/d evidence), so its internals are priced by reference-side
ablations instead (async-reward −5.6 measured; prefix/rr/TMBS rungs staged).

## Beyond F5 (the next frontier)
84% of the F5 step is rollout (33.5 s vs 2.6 s roofline): serial-head removal
(Pathways-compatible multiproc-DP / process-pooled window handling), window 1024 probe,
chunked-logps + HBM re-raise.
