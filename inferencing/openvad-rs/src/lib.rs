// Copyright (c) 2026 Lukas Hirsch
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Python bindings for `openvad-core`.

use pyo3::prelude::*;

#[pymodule]
mod _native {
    use numpy::{PyArray1, PyReadonlyArray1};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    /// One-way group delay of the half-band filter in samples at 16 kHz.
    #[pymodule_export]
    const GROUP_DELAY_16K: usize = openvad_core::GROUP_DELAY_16K;

    /// Computes `input * weight^T + bias`, all row-major and flat.
    // PyO3 extracts Python lists only into owned `Vec`s, never `&[f32]`.
    #[allow(clippy::needless_pass_by_value)]
    #[pyfunction]
    fn linear(input: Vec<f32>, weight: Vec<f32>, bias: Vec<f32>, in_features: usize) -> Vec<f32> {
        openvad_core::linear(&input, &weight, &bias, in_features)
    }

    /// The half-band taps as a new float32 array.
    #[pyfunction]
    fn half_band_taps(py: Python<'_>) -> Bound<'_, PyArray1<f32>> {
        PyArray1::from_slice(py, &openvad_core::HALF_BAND)
    }

    /// Filters 16 kHz audio and keeps every second sample (see `openvad_core::decimate_2x`).
    // PyO3 extracts a NumPy array only by value.
    #[allow(clippy::needless_pass_by_value)]
    #[pyfunction]
    fn decimate_2x<'py>(
        py: Python<'py>,
        a: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_vec(
            py,
            openvad_core::decimate_2x(contiguous(&a)?),
        ))
    }

    /// Doubles the rate of 8 kHz audio (see `openvad_core::interpolate_2x`).
    #[allow(clippy::needless_pass_by_value)]
    #[pyfunction]
    fn interpolate_2x<'py>(
        py: Python<'py>,
        a: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        Ok(PyArray1::from_vec(
            py,
            openvad_core::interpolate_2x(contiguous(&a)?),
        ))
    }

    // A strided view is refused rather than copied, so a caller never pays for a hidden copy.
    fn contiguous<'a>(a: &'a PyReadonlyArray1<'_, f32>) -> PyResult<&'a [f32]> {
        a.as_slice().map_err(|_| {
            PyValueError::new_err("the array must be C-contiguous; pass np.ascontiguousarray(a)")
        })
    }
}
