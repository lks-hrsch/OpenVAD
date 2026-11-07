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
    0xb847b022, 0x39802e22, 0xba10450a, 0x3a787089, 0xbad3cc85, 0x3b20428e, 0xbb7878aa, 0x3bb152c2,
    0xbbfe2e59, 0x3c30cbf3, 0xbc6eb714, 0x3ca0347f, 0xbcd61d4d, 0x3d0e6ee2, 0xbd412601, 0x3d875c9a,
    0xbdcea75b, 0x3e55a9a9, 0xbef749e0, 0xbf22165b, 0x3f4d1dbc, 0x3e20514c, 0x3f2bc631, 0xbe22ad62,
    0xbe1e095a, 0xbd8587ac, 0x3e9284d2, 0xbf281b34, 0xbed94065, 0x3e7863de, 0x3e7f3609, 0x3f50e5be,
    0x3ec4eee4, 0xbe762927, 0x3ac421e6, 0xbe2337b9, 0x3f14622f, 0x3e5634ed, 0xbe43b527, 0xbdab04bc,
];

const INTERPOLATED_FIRST_40: [u32; 80] = [
    0xb8c7b022, 0x00000000, 0x39acf579, 0x00000000, 0xb992d078, 0x00000000, 0x3a57ff3d, 0x00000000,
    0xbac43c78, 0x00000000, 0x3b0ac409, 0x00000000, 0xbb59e559, 0x00000000, 0x3b99901b, 0x00000000,
    0xbbd96a50, 0x00000000, 0x3c102e3d, 0x00000000, 0xbc4416b2, 0x00000000, 0x3c844a9c, 0x00000000,
    0xbcb1fbc7, 0x00000000, 0x3cf2cf00, 0x00000000, 0xbd28d319, 0x00000000, 0x3d7689ee, 0x00000000,
    0xbdc921b0, 0x00000000, 0x3e7b49d1, 0x3e8c3faf, 0xbd6bddd3, 0xbeebbd08, 0xbf34b798, 0xbf6b0585,
    0xbf901229, 0xbf7789b1, 0xbe734726, 0x3f2064f5, 0x3f8182e9, 0x3f5354b3, 0x3ee17ad4, 0x3e5a63dc,
    0x3e8b682d, 0x3eeb0125, 0x3ed0ece7, 0x3db2b01e, 0x3dfb1b8c, 0x3f5ec1d0, 0x3fb47b19, 0x3f21b78f,
    0xbf4787b1, 0xbf7e990f, 0x3e08a43d, 0x3f36fdb1, 0xbe1576e6, 0xbf6ecddf, 0xbebc6280, 0x3eeb2acc,
    0x3df7d694, 0xbf261077, 0xbea815f9, 0x3f39f296, 0x3f6fb197, 0x3da9d33e, 0xbf130574, 0xbecd1853,
    0xbd4d9412, 0xbe1e562a, 0xbf1356ea, 0xbf718016, 0xbf866f45, 0xbf405df1, 0xbe0cb553, 0x3eaeb82b,
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
    assert_eq!(GROUP_DELAY_16K, 35);
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
fn dc_gain_is_one() {
    let ones = [1.0_f32; 200];
    for v in &decimate_2x(&ones)[GROUP_DELAY_16K..] {
        assert!((v - 1.0).abs() < 1e-6, "{v}");
    }
    for v in &interpolate_2x(&ones)[2 * GROUP_DELAY_16K..] {
        assert!((v - 1.0).abs() < 1e-6, "{v}");
    }
}

#[test]
fn known_vector_is_bitwise_stable() {
    assert_eq!(bits(&decimate_2x(&INPUT)), DECIMATED);
    assert_eq!(bits(&interpolate_2x(&INPUT[..40])), INTERPOLATED_FIRST_40);
}
