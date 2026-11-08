# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""One adapter per source, mirroring `openvad_train.fetch`."""

import gzip
from pathlib import Path


def count_lines(path: Path) -> int:
    """Lines of a text or `.gz` file, for the manifest's count of what an adapter skipped."""
    with gzip.open(path, "rb") if path.suffix == ".gz" else path.open("rb") as f:
        return sum(1 for _ in f)
