from pathlib import Path

import safetensors.torch
import torch

from openvad import load_linear


FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/smoke/linear.safetensors"


def test_torch_matches_fixture():
    t = safetensors.torch.load_file(FIXTURE)
    out = load_linear(FIXTURE)(t["input"])
    torch.testing.assert_close(out, t["expected"], atol=1e-6, rtol=0)
