# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from pathlib import Path

import numpy as np
import safetensors.numpy

import openvad_rs


FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/smoke/linear.safetensors"


def test_bindings_match_fixture():
    t = safetensors.numpy.load_file(FIXTURE)
    out = openvad_rs.linear(t["input"].ravel().tolist(), t["weight"].ravel().tolist(), t["bias"].tolist(), 4)
    np.testing.assert_allclose(out, t["expected"].ravel(), atol=1e-6, rtol=0)
