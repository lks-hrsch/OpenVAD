# 0012: Classifier backbone

Status: Proposed

## Context

Candidates: convolution, LSTM, Transformer, Mamba, and combinations.

- Foreground VAD (2026) already compared Mamba / LSTM / Transformer at ~0.6 M params: Mamba ≈ LSTM > Transformer; the backbone is secondary to supervision.
- Causal Mamba matches or beats causal Transformers at small scale on speech enhancement with fewer FLOPs (SEMamba; Zhang et al. 2024).
- Mamba is not more efficient than a Transformer below a threshold sequence length (Speech Slytherin).
- Unidirectional Mamba is clearly worse than bidirectional.
- Mamba training can be unstable on small data (Miyazaki et al. 2024: dropout + AdamW).
- On raw audio waveforms, Mamba's selection hurts compared with time-invariant S4 (Gu & Dao 2023, App. E.4).

Mamba vs. Transformer in short:

- Transformer: attention over all past frames, KV cache grows with context, needs positional encoding, parallel training; streaming only with a bounded window (restricted attention O(r·n·d)).
- Mamba: linear recurrence with input-dependent Δ, B, C (Δ is a generalised RNN gate), constant state, O(1) per frame, parallel scan training. Mamba-2 shows state space models ≈ linear attention with data-dependent decay.
- At VAD scale (D = 64), Mamba carries ~20× more state per stream than an LSTM with the same parameter count (~9.7 KB vs. ~0.5 KB FP32). This is the owner's own estimate.

## Proposal

- Main line: Conv + LSTM/GRU (Silero-style CRNN).
- Mamba as one comparison model in PyTorch (N = 16, E = 2–4, S4D-real init).
- Optionally a linear time-invariant state space model (S4D) vs. selective (Mamba) ablation.
- Transformer only with a causal window, or cited as an offline upper bound.
- Pure conv (MarbleNet/kiloVAD style) as a cheap baseline.
- Only the winner is implemented in Rust ([0011](0011-only-best-model-in-rust.md)).

## Open questions

- Which candidates (Mamba, extra frontends) are explored in PyTorch.

## References

- Vaswani et al. 2017
- Gu, Goel & Ré 2022 (S4)
- Gu & Dao 2023 (Mamba)
- Dao & Gu 2024 (Mamba-2)
- Lahoti et al. 2026 (Mamba-3)
- Chao et al. 2024 (SEMamba)
- Zhang et al. 2024
- Jiang et al. 2024 (Speech Slytherin)
- Miyazaki et al. 2024
- Foreground VAD 2026
- S4VAD 2026
