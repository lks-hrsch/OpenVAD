# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from openvad_train.dataset import __main__ as cli


def test_unimplemented_stage_is_an_error():
    with pytest.raises(SystemExit, match="not yet implemented stage 'align'"):
        cli.main(["align"])


def test_stage_receives_its_arguments(monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setitem(cli.STAGES, "index", seen.append)
    cli.main(["index", "--source", "swc"])
    assert seen == [["--source", "swc"]]
