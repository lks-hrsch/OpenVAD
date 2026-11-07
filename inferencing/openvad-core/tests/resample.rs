// Copyright (c) 2026 Lukas Hirsch
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

// The known vector is pasted as printed, without digit separators.
#![allow(clippy::unreadable_literal)]

use openvad_core::{GROUP_DELAY_16K, HALF_BAND, decimate_2x, interpolate_2x};

const INPUT: [f32; 80] = [
    0.27392337,
    -0.46042657,
    -0.918053,
    -0.96694475,
    0.6265405,
    0.82551116,
    0.21327156,
    0.4589931,
    0.08724998,
    0.87014484,
    0.63170713,
    -0.994523,
    0.7148085,
    -0.93282884,
    0.4593109,
    -0.64868873,
    0.7263578,
    0.082922444,
    -0.4005762,
    -0.15462556,
    -0.9433607,
    -0.75143343,
    0.34124884,
    0.29437903,
    0.23077023,
    -0.23264489,
    0.9944199,
    0.9616707,
    0.37108397,
    0.30091855,
    0.37689346,
    -0.22215715,
    -0.729807,
    0.44297668,
    0.050708644,
    -0.37951624,
    -0.028329283,
    0.77897567,
    0.86808705,
    -0.2844096,
    0.14305966,
    -0.35626122,
    0.18860006,
    -0.32417756,
    -0.216762,
    0.7805487,
    -0.5456848,
    0.2463743,
    -0.8319693,
    0.66528827,
    0.57419664,
    -0.5212611,
    0.7529685,
    -0.88286394,
    -0.32776588,
    -0.6994411,
    -0.09932127,
    0.59264857,
    -0.5387156,
    -0.8959574,
    -0.19089632,
    -0.60297394,
    -0.8184939,
    0.16066477,
    -0.40260774,
    0.34398976,
    -0.60096914,
    0.8842262,
    -0.26977965,
    -0.78900945,
    0.2582163,
    0.8543091,
    -0.11924569,
    0.909181,
    -0.00020837263,
    -0.14954275,
    0.2404269,
    0.990193,
    0.89788735,
    -0.07990972,
];

const DECIMATED: [u32; 40] = [
    0x3892fcb3, 0xb9b9f626, 0x3a4c25a9, 0xbaab5dfa, 0x3b10287b, 0xbb567244, 0x3ba5410f, 0xbbe936f2,
    0x3c267032, 0xbc66a48d, 0x3c9b4615, 0xbcd0fb37, 0x3d0cf40b, 0xbd3f24ab, 0x3d86ecdf, 0xbdce9c13,
    0x3e55b740, 0xbef76d81, 0xbf21e5d7, 0x3f4ce411, 0x3e217035, 0x3f2b6ec9, 0xbe216e61, 0xbe1fd1a5,
    0xbd823c70, 0x3e91b137, 0xbf27a7ca, 0xbeda0801, 0x3e79a965, 0x3e7de5f3, 0x3f511033, 0x3ec49154,
    0xbe75fa6b, 0x3ad1b312, 0xbe239cb3, 0x3f1485bc, 0x3e55360c, 0xbe427558, 0xbdad4511, 0x3e1b84c5,
];

const INTERPOLATED_FIRST_40: [u32; 80] = [
    0x3912fcb3, 0x00000000, 0xb9f9245b, 0x00000000, 0x39c05389, 0x00000000, 0xba95ce21, 0x00000000,
    0x3b06ae5e, 0x00000000, 0xbb396c05, 0x00000000, 0x3b910b21, 0x00000000, 0xbbc9d322, 0x00000000,
    0x3c0e26b7, 0x00000000, 0xbc3ae23f, 0x00000000, 0x3c7ec287, 0x00000000, 0xbcad0121, 0x00000000,
    0x3cebe108, 0x00000000, 0xbd253027, 0x00000000, 0x3d71a309, 0x00000000, 0xbdc6160e, 0x00000000,
    0x3e79ad02, 0x3e8c3faf, 0xbd64dc3a, 0xbeebbd08, 0xbf3531f4, 0xbf6b0585, 0xbf8fcb80, 0xbf7789b1,
    0xbe75c10a, 0x3f2064f5, 0x3f81c6d9, 0x3f5354b3, 0x3ee04585, 0x3e5a63dc, 0x3e8c931d, 0x3eeb0125,
    0x3ecfbe7a, 0x3db2b01e, 0x3dffafc8, 0x3f5ec1d0, 0x3fb43f7e, 0x3f21b78f, 0xbf4711e8, 0xbf7e990f,
    0x3e07b1cc, 0x3f36fdb1, 0xbe14b5c2, 0xbf6ecddf, 0xbebc6d0f, 0x3eeb2acc, 0x3df6aad3, 0xbf261077,
    0xbea7923c, 0x3f39f296, 0x3f6f4d8a, 0x3da9d33e, 0xbf128f16, 0xbecd1853, 0xbd56b456, 0xbe1e562a,
    0xbf12b8e5, 0xbf718016, 0xbf86b733, 0xbf405df1, 0xbe09ce07, 0x3eaeb82b, 0x3eca86f9, 0x3e96b8d9,
];

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// Naive f64 convolution: `out[n] = scale * sum_k HALF_BAND[k] * x[n - k]`, zero before the start.
fn convolve(x: &[f64], scale: f64) -> Vec<f64> {
    (0..x.len())
        .map(|n| {
            scale
                * HALF_BAND
                    .iter()
                    .enumerate()
                    .take(n + 1)
                    .map(|(k, &tap)| f64::from(tap) * x[n - k])
                    .sum::<f64>()
        })
        .collect()
}

fn zero_stuffed(x: &[f32]) -> Vec<f64> {
    x.iter().flat_map(|&v| [f64::from(v), 0.0]).collect()
}

fn assert_close(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((f64::from(*a) - e).abs() < 1e-6, "{a} vs {e}");
    }
}

fn impulse(len: usize, at: usize) -> Vec<f32> {
    let mut x = vec![0.0; len];
    x[at] = 1.0;
    x
}

fn peak(x: &[f32]) -> usize {
    (0..x.len())
        .max_by(|&a, &b| x[a].abs().total_cmp(&x[b].abs()))
        .unwrap()
}

#[test]
fn group_delay_is_the_centre_tap() {
    assert_eq!(HALF_BAND.len(), 2 * GROUP_DELAY_16K + 1);
    assert_eq!(GROUP_DELAY_16K, 33);
}

#[test]
fn interpolate_doubles_length() {
    assert_eq!(interpolate_2x(&[0.0; 100]).len(), 200);
}

#[test]
fn decimate_halves_length() {
    assert_eq!(decimate_2x(&[0.0; 200]).len(), 100);
}

#[test]
fn odd_length_decimation_keeps_the_even_indexed_samples() {
    let x: Vec<f32> = (0..201_u16).map(|i| f32::from(i).sin()).collect();
    let y = decimate_2x(&x);
    assert_eq!(y.len(), 101);
    assert_eq!(bits(&y[..100]), bits(&decimate_2x(&x[..200])));
}

#[test]
fn empty_input_gives_empty_output() {
    assert_eq!(decimate_2x(&[]), Vec::<f32>::new());
    assert_eq!(interpolate_2x(&[]), Vec::<f32>::new());
}

#[test]
fn impulse_response_is_causal_and_delayed() {
    let y = interpolate_2x(&impulse(100, 0));
    for (j, &v) in y.iter().enumerate() {
        assert_eq!(v, 2.0 * HALF_BAND.get(j).copied().unwrap_or(0.0));
    }
    assert_eq!(peak(&y), GROUP_DELAY_16K);
    assert_eq!(y[GROUP_DELAY_16K], 1.0);
    // An impulse at an odd input index lands the centre tap on an even output of the decimator.
    let d = decimate_2x(&impulse(200, 1));
    assert_eq!(peak(&d), GROUP_DELAY_16K.div_ceil(2));
    assert_eq!(d[GROUP_DELAY_16K.div_ceil(2)], 0.5);
}

#[test]
fn matches_an_f64_reference() {
    let x = &INPUT;
    let y: Vec<f64> = convolve(&x.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(), 1.0)
        .into_iter()
        .step_by(2)
        .collect();
    assert_close(&decimate_2x(x), &y);
    assert_close(
        &interpolate_2x(&x[..40]),
        &convolve(&zero_stuffed(&x[..40]), 2.0),
    );
}

#[test]
fn dc_gain_is_within_0_0025_db_of_one() {
    // Half-band symmetry puts DC at a ripple extremum; measured +0.0019 dB.
    let db = |v: f64| (20.0 * v.log10()).abs();
    let ones = [1.0_f32; 200];
    for &v in &decimate_2x(&ones)[GROUP_DELAY_16K..] {
        assert!(db(f64::from(v)) < 0.0025, "{v}");
    }
    // The 8 kHz image of DC (about -67 dB) alternates the output, so take the mean of each pair.
    for pair in interpolate_2x(&ones)[2 * GROUP_DELAY_16K..]
        .as_chunks::<2>()
        .0
    {
        let mean = f64::from(pair[0] + pair[1]) / 2.0;
        assert!(db(mean) < 0.0025, "{pair:?}");
    }
}

#[test]
fn centre_tap_and_zeros_make_interpolation_copy_delayed_input() {
    assert_eq!(HALF_BAND[GROUP_DELAY_16K], 0.5);
    for (k, &tap) in HALF_BAND.iter().enumerate() {
        if k != GROUP_DELAY_16K && k.abs_diff(GROUP_DELAY_16K) % 2 == 0 {
            assert_eq!(tap, 0.0, "tap {k}");
        }
    }
    let y = interpolate_2x(&INPUT[..40]);
    for (m, &x) in INPUT[..40].iter().enumerate() {
        if let Some(&v) = y.get(2 * m + GROUP_DELAY_16K) {
            assert_eq!(v.to_bits(), x.to_bits(), "input {m}");
        }
    }
}

#[test]
fn known_vector_is_bitwise_stable() {
    assert_eq!(bits(&decimate_2x(&INPUT)), DECIMATED);
    assert_eq!(bits(&interpolate_2x(&INPUT[..40])), INTERPOLATED_FIRST_40);
}
