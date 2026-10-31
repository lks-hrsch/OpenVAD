from pathlib import Path

import numpy as np
import safetensors.numpy

from vadigo_train.export_smoke import main


def test_export_writes_expected(tmp_path):
    out = tmp_path / "linear.safetensors"
    main([str(out)])
    t = safetensors.numpy.load_file(out)
    assert {k: v.dtype for k, v in t.items()} == dict.fromkeys(["weight", "bias", "input", "expected"], np.float32)
    assert t["expected"].tolist() == [[6.125, -1.5], [1.0, 0.0], [0.4375, 0.5]]


def test_export_is_byte_identical(tmp_path):
    main([str(tmp_path / "a.safetensors")])
    main([str(tmp_path / "b.safetensors")])
    assert (tmp_path / "a.safetensors").read_bytes() == (tmp_path / "b.safetensors").read_bytes()


def test_export_matches_committed_fixture(tmp_path):
    committed = Path(__file__).resolve().parents[2] / "inferencing/fixtures/smoke/linear.safetensors"
    main([str(tmp_path / "linear.safetensors")])
    assert (tmp_path / "linear.safetensors").read_bytes() == committed.read_bytes()
