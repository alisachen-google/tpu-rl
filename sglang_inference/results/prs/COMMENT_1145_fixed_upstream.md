Comment draft for #1145 (post after the upstream-main verification run finishes):

This is fixed on `main` by PR #1138 ("[TPU/PD] Fix double KV free when aborting a chunked prefill under overlap", merged 2026-10-08): bab49bcec2, with 7366de9460 and
2645aa14af synchronising the deferred drain across ranks. `_retire_aborted_prefill_result` now returns early for a request parked on
`_deferred_chunked_sender_aborts`, which is the single-owner fix proposed above. Verified on our 1-prefill + 1-decode v7x pair at
origin/main (2cec1cf194): 6 client aborts during chunked prefill of a 131k prompt (3 at 2.0 s, 3 at 6.4 s), 0 scheduler exceptions, engines back
to baseline, follow-up requests served. Our independent branch `pd-chunked-abort-single-release` is superseded and will not be opened as a PR.
Closing.
