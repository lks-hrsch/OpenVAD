# 0017: Experiment budget

Status: Proposed

## Context

The full grid (windows × representations × backbones × sample rates × formats × seeds) is several hundred runs, but week 10 is the only slot for variants.

## Proposal

- Vary one factor at a time around a baseline (STFT/log-mel → conv → LSTM/GRU, 32/16 ms, main sample rate, FP32), ≈ 20 configs.
- Screen with 1 seed; use 3 seeds only for winners and headline results.
- Number formats only on the final model.

## Open questions

- The scope; the owner decides it.
