# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import os

import safetensors.torch
import torch


def load_linear(path: str | os.PathLike[str]) -> torch.nn.Linear:
    t = safetensors.torch.load_file(path)
    layer = torch.nn.Linear(t["weight"].shape[1], t["weight"].shape[0])
    with torch.no_grad():
        layer.weight.copy_(t["weight"])
        layer.bias.copy_(t["bias"])
    return layer
