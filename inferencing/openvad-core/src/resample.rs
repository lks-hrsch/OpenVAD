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
//!
//! - The functions are stateless whole-signal calls. Calling them per block
//!   restarts every block from silence and gives a different result without an
//!   error, so pass the whole signal.
//! - Output can leave [-1, 1] (a full-scale square wave overshoots by about
//!   14%), so a caller that casts to int16 must clip.

pub use crate::resample_coefficients::HALF_BAND;

/// One-way group delay of the filter in samples at 16 kHz.
///
/// Each of [`decimate_2x`] and [`interpolate_2x`] delays by this much, so the
/// narrowband round trip (16 kHz to 8 kHz and back) delays by twice this. At
/// 8 kHz the delay is `GROUP_DELAY_16K / 2` samples, a half-integer for the
/// odd value, so integer division by 2 is wrong.
pub const GROUP_DELAY_16K: usize = (HALF_BAND.len() - 1) / 2;

/// Filters at 16 kHz and keeps the output at input indices 0, 2, 4, ... .
///
/// Output `i` is the filtered signal at input index `2 * i`, so it describes
/// input time `2 * i - GROUP_DELAY_16K` (in 16 kHz samples). The output has
/// `input.len().div_ceil(2)` samples: an odd-length input keeps its last sample's
/// position. DC gain is 1 within 0.0025 dB, and empty input gives empty output.
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
/// end of the input is dropped. After the start-up transient the mean of each
/// output pair has DC gain 1 within 0.0025 dB; the two outputs of a pair differ
/// by about 4e-4 because of the 8 kHz image. Empty input gives empty output.
///
/// The taps that would multiply an inserted zero are left out of the sum. For
/// finite input this equals filtering the zero-stuffed signal bit for bit.
///
/// A round trip `interpolate_2x(&decimate_2x(x))` has length `x.len()` for even
/// `x.len()` and `x.len() + 1` for odd, and delays by `2 * GROUP_DELAY_16K`.
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
