# 0005: Pipeline stages

Status: Proposed

## Context

A VAD pipeline has the stages (optional denoising) → signal representation → classifier → post-processing (threshold, hysteresis, minimum speech/silence duration, hangover).
A denoising front end adds latency and compute; the VAD can learn robustness directly (Braun & Tashev 2021).

## Proposal

- Post-processing is part of the evaluated system and is tuned on dev only.
- No denoising front end.

## References

- Braun & Tashev 2021 (VNR, BCE+MAE)
