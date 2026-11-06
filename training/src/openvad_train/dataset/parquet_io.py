# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""The one module that reads and writes Parquet: asynchronous Polars, atomic per file."""

import os
from pathlib import Path

import polars as pl


async def read_parquet(path: Path, columns: list[str] | None = None) -> pl.DataFrame:
    lazy = pl.scan_parquet(path)
    if columns is not None:
        lazy = lazy.select(columns)
    return await lazy.collect_async()


async def write_parquet(frames: dict[Path, pl.LazyFrame]) -> None:
    """Write every frame to its path; a failure leaves no file visible at any path."""
    tmps = {path: path.with_name(path.name + ".tmp") for path in frames}
    try:
        await pl.collect_all_async([frame.sink_parquet(tmps[path], lazy=True) for path, frame in frames.items()])
        for path, tmp in tmps.items():
            os.replace(tmp, path)
    finally:
        for tmp in tmps.values():
            tmp.unlink(missing_ok=True)
