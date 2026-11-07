from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

GROUP_DELAY_16K: int

def linear(
    input: Sequence[float],
    weight: Sequence[float],
    bias: Sequence[float],
    in_features: int,
) -> list[float]: ...
def half_band_taps() -> NDArray[np.float32]: ...
def decimate_2x(a: NDArray[np.float32]) -> NDArray[np.float32]: ...
def interpolate_2x(a: NDArray[np.float32]) -> NDArray[np.float32]: ...
