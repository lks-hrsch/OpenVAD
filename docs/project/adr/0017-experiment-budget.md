# 0017: Experiment budget

Status: Proposed

## Context

The full grid (windows × representations × backbones × number formats × seeds) is several hundred runs, but week 10 is the only slot for variants.
A native 8 kHz model is an optional extension ([0009](0009-native-16khz-and-one-resampler.md)), not a grid axis.
Ablations are proposed in [0006](0006-window-and-stride.md), [0010](0010-signal-representation.md) and [0012](0012-classifier-backbone.md).

## Proposal

- Vary one factor at a time around a baseline (STFT/log-mel → conv → LSTM/GRU, 16 kHz, FP32), ≈ 20 configs. Window and stride follow [0006](0006-window-and-stride.md), which is Open; 32/16 ms is the proposed window and stride, open in 0006.
- Stages (owner, 2026-11-04):
  - LibriVAD `small` train and OpenVAD Small compare architectures, run separately (owner, 2026-11-04) and never mixed (owner, 2026-11-03);
  - OpenVAD Small also finds hyperparameters;
  - OpenVAD Large is the full training.
- Screen with 1 seed; use 3 seeds only for winners and headline results, on the Large tier.
- There will be a Small and a Large set, each with a train, validation (dev) and test split, and Small is a subset of Large ([0023](0023-dataset-composition.md)) (owner, 2026-11-06).
- Running on LibriVAD `small` train makes architectures comparable with the LibriVAD paper models. LibriVAD test splits are never trained on: `medium` test is a benchmark and `small` test is the paper-comparison row ([0020](0020-evaluation-benchmarks.md)). LibriVAD `medium` train reaches training only through Large (owner, 2026-11-04).
- LibriVAD has no validation split at the pinned revision, so screening validates on the OpenVAD dev split.
- Early stopping and checkpoint choice use a fixed subset of 15 to 30 h of dev.
- Retune the 2 or 3 best configurations at Large scale (learning rate, width, epochs) before their seeds; a scaling check between the tiers is optional.
- Compare runs by steps or frames seen, not by epochs.
- Number formats only on the final model.

## Open questions

- What does the budget cap: the number of configurations per axis, the number of runs, or GPU-hours? Only about 20 configurations are fixed so far.
- Is week 10 the variants slot, and where is the work plan?
- Were the LibriVAD paper models trained on `small` train?
- Where does the optional native 8 kHz model ([0009](0009-native-16khz-and-one-resampler.md)) fit in the budget?
