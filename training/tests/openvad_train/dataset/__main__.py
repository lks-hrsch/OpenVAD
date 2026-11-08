# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import pytest

from openvad_train.dataset import __main__ as cli
from openvad_train.dataset import align, construct, handcheck, index, publish, verify
from openvad_train.dataset.config import PILOT, SLICE


def test_unimplemented_stage_is_an_error():
    with pytest.raises(SystemExit, match="not yet implemented stage 'pilot'"):
        cli.main(["pilot"])


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


def test_verify_is_registered():
    assert cli.STAGES["verify"] is verify.main


def test_align_is_registered_and_takes_a_config_and_languages(monkeypatch):
    seen = []
    monkeypatch.setattr(align, "run_align", lambda c, languages, **kw: seen.append((c.name, languages, kw)))
    cli.main(["align", "--config", "slice", "--languages", "de,fr"])
    cli.main(["align", "--config", "full"])
    none = {"max_clips": None}
    assert seen == [(SLICE.name, ["de", "fr"], none), ("openvad-v1", None, none)]


def test_align_takes_the_pilot_with_a_clip_limit(monkeypatch):
    seen = []
    monkeypatch.setattr(align, "run_align", lambda c, languages, **kw: seen.append((c, languages, kw)))
    cli.main(["align", "--config", "pilot", "--languages", "ca", "--max-clips", "200"])
    assert seen == [(PILOT, ["ca"], {"max_clips": 200})]


def test_index_and_the_hand_check_take_the_pilot(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(index, "run_index", seen.append)
    monkeypatch.setattr(handcheck, "sample_handcheck", lambda c, n, languages: seen.append(c) or tmp_path)
    monkeypatch.setattr(handcheck, "score_handcheck", lambda path: seen.append(path) or {})
    cli.main(["index", "--config", "pilot"])
    cli.main(["handcheck-sample", "--config", "pilot"])
    cli.main(["handcheck-score", "--config", "pilot"])
    assert seen == [PILOT, PILOT, handcheck.handcheck_dir(PILOT)]


def test_the_hand_check_stages_are_registered():
    assert cli.STAGES["handcheck-sample"] is handcheck.main_sample
    assert cli.STAGES["handcheck-score"] is handcheck.main_score
