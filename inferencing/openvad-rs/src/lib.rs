// Copyright (c) 2026 Lukas Hirsch
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Python bindings for `openvad-core`.

use pyo3::prelude::*;

#[pymodule]
mod _native {
    use pyo3::prelude::*;

    /// Computes `input * weight^T + bias`, all row-major and flat.
    // PyO3 extracts Python lists only into owned `Vec`s, never `&[f32]`.
    #[allow(clippy::needless_pass_by_value)]
    #[pyfunction]
    fn linear(input: Vec<f32>, weight: Vec<f32>, bias: Vec<f32>, in_features: usize) -> Vec<f32> {
        openvad_core::linear(&input, &weight, &bias, in_features)
    }
}
