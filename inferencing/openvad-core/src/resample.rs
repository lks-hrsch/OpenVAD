// Copyright (c) 2026 Lukas Hirsch
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The one 2:1 half-band resampler of ADR 0009 (scalar tier, offline).
//!
//! Both directions are causal and linear-phase with the same [`HALF_BAND`] taps,
//! so a filtered signal lags its input by [`GROUP_DELAY_16K`] samples at the
//! 16 kHz side. Each output is one `mul_add` chain over the taps in ascending
//! order from a zero accumulator (ADR 0026); samples before the start are zero.

pub use crate::resample_coefficients::HALF_BAND;

/// Group delay of the filter in samples at 16 kHz.
pub const GROUP_DELAY_16K: usize = (HALF_BAND.len() - 1) / 2;

/// Filters at 16 kHz and keeps the output at input indices 0, 2, 4, ... .
///
/// Output `i` is the filtered signal at input index `2 * i`, so it describes
/// input time `2 * i - GROUP_DELAY_16K` (in 16 kHz samples). The output has
/// `input.len().div_ceil(2)` samples: an odd-length input keeps its last sample's
/// position. DC gain is 1, and empty input gives empty output.
#[must_use]
pub fn decimate_2x(input: &[f32]) -> Vec<f32> {
    (0..input.len())
        .step_by(2)
        .map(|n| {
            let mut acc = 0.0_f32;
            for (k, &tap) in HALF_BAND.iter().enumerate().take(n + 1) {
                acc = tap.mul_add(input[n - k], acc);
            }
            acc
        })
        .collect()
}

/// Inserts a zero after every sample and filters at 16 kHz with a gain of 2.
///
/// The output has exactly `2 * input.len()` samples, so the filter tail past the
/// end of the input is dropped. DC gain is 1 after the start-up transient, and empty
/// input gives empty output. Zero-valued samples are skipped, which does not change
/// any result.
#[must_use]
pub fn interpolate_2x(input: &[f32]) -> Vec<f32> {
    (0..2 * input.len())
        .map(|n| {
            let mut acc = 0.0_f32;
            for k in (n % 2..HALF_BAND.len().min(n + 1)).step_by(2) {
                acc = HALF_BAND[k].mul_add(input[(n - k) / 2], acc);
            }
            // The factor 2 restores the amplitude lost to the inserted zeros; it is exact.
            2.0 * acc
        })
        .collect()
}
