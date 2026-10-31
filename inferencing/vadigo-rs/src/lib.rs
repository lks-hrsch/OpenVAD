//! Python bindings for `vadigo-core`.

use pyo3::prelude::*;

#[pymodule]
mod _native {
    use pyo3::prelude::*;

    /// Computes `input * weight^T + bias`, all row-major and flat.
    // PyO3 extracts Python lists only into owned `Vec`s, never `&[f32]`.
    #[allow(clippy::needless_pass_by_value)]
    #[pyfunction]
    fn linear(input: Vec<f32>, weight: Vec<f32>, bias: Vec<f32>, in_features: usize) -> Vec<f32> {
        vadigo_core::linear(&input, &weight, &bias, in_features)
    }
}
