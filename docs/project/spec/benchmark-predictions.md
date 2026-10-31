# Benchmark prediction spec

Every system except pyannote runs causally through one common streaming interface: the harness pushes the same fixed chunk into every adapter, each adapter buffers internally to its model's native call size, and outputs are mapped onto the 10 ms frame grid by causal hold. Systems exchange data through files only.

pyannote is non-causal. It is labelled as such everywhere it appears and serves as an upper bound only.

## Input

- A manifest listing the audio files.
- A run config with the sample rate and the chunk size.

## Output

One Parquet file per audio file, with columns:

| Column | Type | Meaning |
|---|---|---|
| `frame` | int | 10 ms frame index |
| `speech_prob` | float32 | speech probability for that frame |

One `meta.json` per run, with the keys `system`, `version`, `weights_sha256`, `runtime`, `threads` and `cpu_model`.
