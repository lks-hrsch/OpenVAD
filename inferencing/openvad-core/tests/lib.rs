use openvad_core::{Error, linear, load_f32};
use safetensors::{Dtype, serialize, tensor::TensorView};

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../fixtures/smoke/linear.safetensors"
);

#[test]
fn linear_matches_pytorch_fixture() {
    let bytes = std::fs::read(FIXTURE).unwrap();
    let (_, w) = load_f32(&bytes, "weight").unwrap();
    let (_, b) = load_f32(&bytes, "bias").unwrap();
    let (shape, x) = load_f32(&bytes, "input").unwrap();
    let (expected_shape, expected) = load_f32(&bytes, "expected").unwrap();
    // Literals: a decoding bug hits both sides, and big-endian turns every value into about 0, inside the tolerance.
    assert_eq!(expected_shape, [3, 2]);
    assert_eq!(expected, [6.125, -1.5, 1.0, 0.0, 0.4375, 0.5]);
    let out = linear(&x, &w, &b, shape[1]);
    assert_eq!(out.len(), expected.len());
    for (o, e) in out.iter().zip(&expected) {
        assert!((o - e).abs() <= 1e-6, "{o} vs {e}");
    }
}

// A weight file exported as float64 must not decode as twice as many garbage floats.
#[test]
fn load_f32_rejects_a_tensor_of_another_dtype() {
    let data = [0u8; 16];
    let view = TensorView::new(Dtype::F64, vec![2], &data).unwrap();
    let bytes = serialize([("w", view)], None).unwrap();
    assert!(matches!(load_f32(&bytes, "w"), Err(Error::NotF32(name)) if name == "w"));
}
