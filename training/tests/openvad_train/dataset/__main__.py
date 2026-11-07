# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from openvad_train.dataset import __main__ as cli
from openvad_train.dataset import construct, publish
from openvad_train.dataset.config import SLICE


def test_unimplemented_stage_is_an_error():
    with pytest.raises(SystemExit, match="not yet implemented stage 'align'"):
        cli.main(["align"])


def test_stage_receives_its_arguments(monkeypatch):
    seen: list[list[str]] = []
    monkeypatch.setitem(cli.STAGES, "index", seen.append)
    cli.main(["index", "--source", "swc"])
    assert seen == [["--source", "swc"]]


def test_the_stage_is_registered_and_passes_its_arguments(monkeypatch):
    seen = []
    monkeypatch.setattr(construct, "run_construct", lambda c, workers: seen.append((c.name, workers)))
    cli.main(["construct", "--config", "slice", "--workers", "3"])
    assert seen == [(SLICE.name, 3)]


def test_workers_default_to_eleven(monkeypatch):
    seen = []
    monkeypatch.setattr(construct, "run_construct", lambda c, workers: seen.append((c.name, workers)))
    cli.main(["construct", "--config", "full"])
    assert seen == [("openvad-v1", 11)]


def test_publish_is_registered_and_takes_a_config(monkeypatch):
    seen = []
    monkeypatch.setattr(publish, "run_publish", lambda c: seen.append(c.name))
    cli.main(["publish", "--config", "slice"])
    assert seen == [SLICE.name]
