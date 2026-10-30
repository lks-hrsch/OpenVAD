# VADigo

A reproducible, real-time voice activity detector (VAD) for noisy telephone audio.

> 🚧 Work in progress – part of a Master's thesis at TU Dresden
>
> Early development: nothing is usable yet. Code will be published in this repository from the first weeks onwards.

## Motivation

Popular lightweight VADs such as Silero VAD and TEN VAD publish their model weights only. Their architecture, training recipe and training data are not open, which makes them hard to reproduce or build upon.

VADigo aims to be fully reproducible: open architecture, open training code, public training data only, and open evaluation. The goal is not to beat Silero VAD. It is to show how far a fully open VAD can go, and why.

VADigo is developed as part of the Master's thesis "Reproducible Real-Time Voice Activity Detection under Noisy Telephone Conditions" at TU Dresden.

## Goals

- Streaming, real-time operation, lightweight enough for CPU inference
- Focus on narrowband (8 kHz) and wideband (16 kHz) telephone and voice-call audio in noisy conditions
- Per-frame outputs: speech probability, a calibrated confidence, and an estimated signal-to-noise ratio (SNR)
- Training on public datasets only
- A neutral benchmark against existing VADs

## Planned architecture

- **Training and evaluation:** Python
- **Inference:** hand-implemented in Rust, loading the trained weights directly (no ONNX runtime)
- **Python bindings** for the Rust inference engine

## Name origin

The vadigo (*Campogramma glaycos*) is a Mediterranean fish. Its official FAO species code happens to be "VAD".

## License

To be decided (a permissive open-source license is planned).
