use vadigo_core::{linear, load_f32};

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
    let (_, expected) = load_f32(&bytes, "expected").unwrap();
    let out = linear(&x, &w, &b, shape[1]);
    assert_eq!(out.len(), expected.len());
    for (o, e) in out.iter().zip(&expected) {
        assert!((o - e).abs() <= 1e-6, "{o} vs {e}");
    }
}

#[test]
fn load_f32_rejects_missing_tensor() {
    let bytes = std::fs::read(FIXTURE).unwrap();
    assert!(load_f32(&bytes, "nope").is_err());
}
