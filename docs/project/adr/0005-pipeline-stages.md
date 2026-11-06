# 0005: Denoising and post-processing

Status: Proposed

## Context

A VAD pipeline has the stages (optional denoising) → signal representation → classifier → post-processing (threshold, hysteresis, minimum speech/silence duration, hangover).
A denoising front end adds latency and compute; the VAD can learn robustness directly (Braun & Tashev 2021).
The benchmark output is `speech_prob` per 10 ms frame ([spec](../spec/benchmark-predictions.md)).

## Proposal

- No denoising front end.
- Post-processing is not part of the benchmark, also not for the reference VADs: every system is scored on its raw `speech_prob` against a threshold from dev (owner, 2026-11-04).
- If post-processing is used, its parameters are tuned on dev only.

## Consequences

- Robustness comes from the training data: noise augmentation ([0023](0023-dataset-composition.md)) and telephone channel simulation ([0021](0021-telephone-channel-simulation.md)).

## Open questions

- Where does post-processing run (in the engine, in the adapter, or outside the thesis)?
- Are hysteresis, minimum durations and hangover in scope at all, given that the metrics use raw probabilities and one threshold?

## References

- Braun & Tashev 2021
