# 0006: Window and stride

Status: Open

## Context

The window and stride of the signal representation set accuracy, latency and compute, and the harness and the engine depend on them.
The grid under consideration: window 30 ms with stride 10 / 15 / 30 ms; window 32 ms with stride 8 / 16 / 32 ms.

## Findings

- There are three different "windows": the STFT analysis frame, the model chunk per call (Silero: 512 samples at 16 kHz / 256 at 8 kHz = 32 ms; TEN: hop 160 or 256), and the 10 ms evaluation grid. Each experiment must say which one varies.
- For a stateful model, context comes from the state, not the chunk; kiloVAD finds AUC levels off at ~200 ms context.
- The overlaps differ between the two series (30 ms: 67/50/0 %, 32 ms: 75/50/0 %). For a clean ablation, use window × overlap {0, 50, 75 %}.
- Compute scales with 1/stride. 32 ms is a power of two at 8 and 16 kHz (256/512 samples); 30 ms needs zero padding for the fast Fourier transform.
- Strides of 8/15/16/32 ms do not fall on the 10 ms grid; outputs are mapped onto it by causal hold ([spec](../spec/benchmark-predictions.md)).
- The engine buffers internally up to the model's stride ([0009](0009-native-16khz-and-one-resampler.md)), so the stride sets the minimum latency.
- Candidate to add: 20 ms × {10, 20} ms (aligned with 20 ms RTP/G.711 packets, verify).
- The production voicebot feeds its VAD 512-sample blocks at 16 kHz (32 ms); see [build-and-hardware.md](../../notes/build-and-hardware.md).

## Open questions

- Which "window" the window/stride ablation varies.
- The window/stride grid, and whether 32/16 ms is the baseline ([0017](0017-experiment-budget.md) does not treat it as settled).
- Whether the 20 ms candidate is wanted.
- The harness chunk size, decided together with the grid.

## References

- kiloVAD 2026
- Window Size Matters 2026
