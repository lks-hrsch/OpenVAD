# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Write the Linear(4, 2) parity fixture shared by the PyTorch and Rust sides."""

import sys

import numpy as np
from safetensors.numpy import save_file


# Values are exact in float32, so the bytes match on every platform.
WEIGHT = [[0.5, -0.25, 1.0, 0.75], [-1.0, 0.5, 0.25, -0.5]]
BIAS = [0.125, -0.25]
INPUT = [[1.0, 2.0, 3.0, 4.0], [-1.0, 0.5, 0.0, 2.0], [0.25, -0.75, 1.5, -2.0]]


def main(argv: list[str] | None = None) -> None:
    (out,) = sys.argv[1:] if argv is None else argv
    w, b, x = (np.array(v, dtype=np.float32) for v in (WEIGHT, BIAS, INPUT))
    expected = (x.astype(np.float64) @ w.astype(np.float64).T + b).astype(np.float32)
    save_file({"weight": w, "bias": b, "input": x, "expected": expected}, out)


if __name__ == "__main__":
    main()
