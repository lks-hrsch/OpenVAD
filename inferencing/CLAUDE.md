# inferencing

- Parity tests must pass before any change to the model or the engine is done.
- The tolerance is an absolute `1e-6` ([ADR 0002](../docs/project/adr/0002-safetensors-weights.md)).
- Run `just fixtures` to regenerate the parity fixture and `just check-fixtures` to fail on a stale one.
- Keep the engine free of ONNX Runtime, tract, burn and candle; `just guard-engine` checks it.
