# benchmarking

- Each adapter is its own uv project under `benchmarking/adapters/<system>/` and is not a workspace member ([ADR 0003](../docs/project/adr/0003-isolated-benchmark-adapters.md)).
- Adapters talk through files, never imports. The format is in the [benchmark prediction spec](../docs/project/spec/benchmark-predictions.md).
- "Identical chunks": the harness pushes the same fixed chunk into every adapter and each adapter buffers internally to its model's native call size. Outputs off the 10 ms grid are mapped by causal hold. Algorithmic latency is reported per system.
- Resampled audio comes from the offline data preparation script (the 8 and 16 kHz conversions use the one OpenVAD resampler via `openvad-rs`; other rates use ffmpeg, [ADR 0009](../docs/project/adr/0009-native-16khz-and-one-resampler.md)). Adapters read prepared files and never import `openvad-rs`. The upsampling per system is recorded in the metadata.
- ONNX Runtime is allowed here only.
- The TEN VAD adapter stays isolated: it has extra non-OSI conditions (Agora non-compete). Document its licence in the adapter.
- Never redistribute or commit third-party weights or data. Scripts fetch them.
