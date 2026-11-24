# q30-lab-round002

A specialized inference engine for [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B) at revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`, built with EngineSmith 0.3.0.dev0.

- Machine: 16 x Trainium (trn2.48xlarge), 64 cores, 24 GiB, Ubuntu 24.04.5 LTS
- Reference: nxdi, tensor-parallel degree 4 (the engine gets as many accelerator cores)
- Judged under evaluation version 2
- Decode: not recorded (this run predates the one-forward-per-token rule)
- Source and data: https://github.com/xuanfeiren/enginesmith-q30-lab-round002
- Search: 8 rounds by Claude Code 2.1.317 (Claude Code) (model us.anthropic.claude-opus-5-5, reasoning effort medium), agent cost $329.02

## Notes from the author

- This run predates the held-out final evaluation and the one-forward-per-token rule (evaluation version 2), so the numbers below come from the search evaluation.
- Round 002 is the best round of this run that generates one token per forward pass. From round 003 on, the agent turned to speculative decoding, whose speed on the evaluation's random-token prompts overstates what it gives on real text, so those rounds (and their scores in the table below) are not what is shared here.
- A clean rebuild of round 002, checked later with the current procedure on fresh held-out prompts (5 rounds, the engine given only the reference's tensor-parallel degree in cores): passed; score 2.88x NxD Inference. Decode probe 1.06.

## Speed against nxdi (search evaluation)

Score 2.91x; prefill 3.69x, decode 2.30x (reference time / engine time, 65 tokens per request; the fraction of the hardware bound is bound time / engine time, an estimate).

| prompt tokens | prefill | decode | prefill, fraction of the bound | decode, fraction of the bound |
|---:|---:|---:|---:|---:|
| 256 | 4.90x | 2.23x | not computed | not computed |
| 1,024 | 4.23x | 2.41x | not computed | not computed |
| 4,096 | 2.42x | 2.27x | not computed | not computed |

## Speed in absolute terms

Medians over the paired rounds of the search evaluation, this engine and nxdi on the same machine.

| prompt tokens | time to first token, ms | prefill tok/s | decode tok/s |
|---:|---:|---:|---:|
| 256 | 68.5 (nxdi 335.5) | 3,737.6 (763.1) | 203.2 (91.0) |
| 1,024 | 105.6 (nxdi 446.9) | 9,695.2 (2,291.2) | 202.4 (83.9) |
| 4,096 | 284.6 (nxdi 689.3) | 14,393.8 (5,942.6) | 193.4 (85.3) |

## Checks on the decode speed

- Natural text: not measured (no task accuracy answers).
- Decode probe: 64 one-token requests took 1.06x as long as one request for the same tokens at 256 prompt tokens (332.6 ms against 313.6 ms; each request 5.06 to 7.41 ms). One forward pass per generated token keeps this near 1; speculative decoding raises it.

## How the search went

- 9 coding sessions by Claude Code 2.1.317 (Claude Code) (model us.anthropic.claude-opus-5-5, reasoning effort medium); the search took 2192 min.
- First correct engine: round 001, 60 min after the start.
- First engine as fast as nxdi (score 1.0): round 001, 60 min after the start.
- Agent usage: 2,682 input and 1,178k output tokens, 300,932k cache reads, $329.02 (as the agent reported them).

| round | kind | outcome | ended, min | score | prefill | decode | tokens in / out | engine lines |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 001 | generate | accepted (first_passing_candidate) | 59.5 | 2.64x | 3.40x | 2.05x | 114 / 59k | 5,689 |
| 002 | optimize | accepted (score_improved) | 267.7 | 2.91x | 3.69x | 2.30x | 300 / 135k | 6,133 |
| 003 | optimize | accepted (score_improved) | 756.9 | 3.95x | 3.78x | 4.13x | 454 / 239k | 7,365 |
| 004 | optimize | accepted (score_improved) | 1,140.7 | 4.15x | 4.24x | 4.06x | 502 / 239k | 7,632 |
| 005 | optimize | accepted (score_improved) | 1,448.0 | 5.79x | 5.24x | 6.40x | 366 / 176k | 7,808 |
| 006 | optimize | rejected (insufficient_improvement) | 1,905.7 | 5.68x | 5.23x | 6.16x | 486 / 175k | 7,869 |
| 007 | optimize | rejected (insufficient_improvement) | 2,039.3 | 5.49x | 5.47x | 5.51x | 252 / 112k | 7,869 |
| 008 | optimize | rejected (insufficient_improvement) | 2,191.9 | 4.39x | 5.49x | 3.51x | 208 / 41k | 7,879 |

`history.csv` in https://github.com/xuanfeiren/enginesmith-q30-lab-round002 has every round with every prompt length; `enginesmith.json` has the same and more.

## Use it

These are the author's measurements. Check them on a machine like this one (16 x Trainium (trn2.48xlarge); this builds the engine from `source/` and compares it with your local nxdi):

```sh
hf download Qwen/Qwen3-30B-A3B --revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 --local-dir models/Qwen3-30B-A3B
uv run enginesmith verify https://github.com/xuanfeiren/enginesmith-q30-lab-round002 --model models/Qwen3-30B-A3B
```

Build it yourself: `bash build.sh` in `source/`, then run `target/engine --model MODEL_DIR`; it speaks the engine protocol of EngineSmith's `docs/protocol.md` (token ids in, greedy token ids out). Read the source before you run it: it is native code written by a coding agent.

`enginesmith.json` holds every setting and result, and the SHA-256 of each model file.

## License

MIT, see `LICENSE`. Code adapted from other projects keeps its own license and notices (`source/THIRD_PARTY.md` lists it when the agent recorded any).
