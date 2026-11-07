// Copyright (c) 2026 Lukas Hirsch
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Hand-written inference primitives for `OpenVAD`.

use std::fmt;

use safetensors::{Dtype, SafeTensors};

mod resample;
mod resample_coefficients;

pub use resample::{GROUP_DELAY_16K, HALF_BAND, decimate_2x, interpolate_2x};

/// Error returned by [`load_f32`].
#[derive(Debug)]
pub enum Error {
    /// The safetensors bytes or the tensor lookup were invalid.
    SafeTensors(safetensors::SafeTensorError),
    /// The named tensor exists but is not `float32`.
    NotF32(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SafeTensors(e) => write!(f, "safetensors error: {e}"),
            Self::NotF32(name) => write!(f, "tensor `{name}` is not float32"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SafeTensors(e) => Some(e),
            Self::NotF32(_) => None,
        }
    }
}

impl From<safetensors::SafeTensorError> for Error {
    fn from(e: safetensors::SafeTensorError) -> Self {
        Self::SafeTensors(e)
    }
}

/// Reads the `float32` tensor `name` from safetensors `bytes`.
///
/// Returns its shape and its row-major values.
///
/// # Errors
///
/// Returns [`Error::SafeTensors`] if `bytes` is not a valid safetensors file or
/// has no tensor `name`, and [`Error::NotF32`] if the tensor is not `float32`.
pub fn load_f32(bytes: &[u8], name: &str) -> Result<(Vec<usize>, Vec<f32>), Error> {
    let tensors = SafeTensors::deserialize(bytes)?;
    let view = tensors.tensor(name)?;
    if view.dtype() != Dtype::F32 {
        return Err(Error::NotF32(name.to_owned()));
    }
    // safetensors checks that the byte length matches shape and dtype, so no remainder.
    let (chunks, _) = view.data().as_chunks::<4>();
    let values = chunks.iter().map(|c| f32::from_le_bytes(*c)).collect();
    Ok((view.shape().to_vec(), values))
}

/// Computes `input * weight^T + bias`, all row-major.
///
/// `input` is `rows * in_features`, `weight` is `out_features * in_features`
/// and `out_features` is `bias.len()`. The result is `rows * out_features`.
///
/// # Panics
///
/// Panics if `in_features` is zero or the slice lengths do not match.
#[must_use]
pub fn linear(input: &[f32], weight: &[f32], bias: &[f32], in_features: usize) -> Vec<f32> {
    assert!(in_features > 0, "in_features must be positive");
    assert_eq!(input.len() % in_features, 0, "input length mismatch");
    assert_eq!(
        weight.len(),
        bias.len() * in_features,
        "weight length mismatch"
    );
    input
        .chunks_exact(in_features)
        .flat_map(|row| {
            weight
                .chunks_exact(in_features)
                .zip(bias)
                .map(move |(w, b)| row.iter().zip(w).map(|(x, w)| x * w).sum::<f32>() + b)
        })
        .collect()
}
