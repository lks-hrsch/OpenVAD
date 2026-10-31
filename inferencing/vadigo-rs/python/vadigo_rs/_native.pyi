from collections.abc import Sequence

def linear(
    input: Sequence[float],
    weight: Sequence[float],
    bias: Sequence[float],
    in_features: int,
) -> list[float]: ...
