## Live validation on a TPU v7x P/D pair (as requested)

Head 8bbf44e392 deployed on both hosts (1 prefill + 1 decode, gpt-oss-120b, raiden, mini-LB, Config A: radix off, 8 slots). Control = upstream main. Requests go through the mini-LB `/generate` unless noted.

| Case | main | this PR |
|---|---|---|
| `n=2`, 32 tokens | no response; the client gave up after 90 s (the #1147 hang) | **2 samples in 0.3 s** |
| `n=3` | – | 3 samples in 0.3 s |
| `n=8` | – | 8 samples in 1.2 s |
| batch of 2 prompts, `n=2` | – | 4 samples (2 per prompt) in 1.1 s |
| batch of 3 prompts, `n=3` | – | 9 samples in 0.6 s |
| 5 consecutive `n=2` requests (room reuse) | – | 2 samples each, 0.1 s each |
| `/v1/chat/completions` `n=2` | – | 2 choices |
| `n=1` (unchanged path) | – | 1 sample |
| prefix request skipped on the pair | – | the prefill scheduled **41** sequences for 39 sub-requests + 2 warm-ups, and saw **0** zero-token requests; with the prefix request it would have scheduled one more per `n>1` call |
| after the run | – | decode `num_used_tokens` **0**, 0 scheduler exceptions on either engine, both `/health` 200 |
| monolithic server on this head (non-PD path) | – | `n=2` → 2 samples in 1.3 s, `n=3` → 3 samples in 0.9 s, 0 exceptions (the prefix request is still sent off the pair) |

Every sample produced the requested number of tokens. The behaviour matches what we measured for #1192 (our fix for the same issue); the two PRs take equivalent approaches (per-sample room from the normalised block vs. `room + k * batch_size`, both skipping the prefix request in PD mode), so either one closes #1147.

