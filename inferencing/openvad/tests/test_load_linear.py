# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from pathlib import Path

import safetensors.torch
import torch

from openvad import load_linear


FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/smoke/linear.safetensors"


def test_torch_matches_fixture():
    t = safetensors.torch.load_file(FIXTURE)
    out = load_linear(FIXTURE)(t["input"])
    torch.testing.assert_close(out, t["expected"], atol=1e-6, rtol=0)
