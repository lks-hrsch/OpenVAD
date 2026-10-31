from pathlib import Path

import numpy as np
import safetensors.numpy

import vadigo_rs


FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/smoke/linear.safetensors"


def test_bindings_match_fixture():
    t = safetensors.numpy.load_file(FIXTURE)
    out = vadigo_rs.linear(t["input"].ravel().tolist(), t["weight"].ravel().tolist(), t["bias"].tolist(), 4)
    np.testing.assert_allclose(out, t["expected"].ravel(), atol=1e-6, rtol=0)
