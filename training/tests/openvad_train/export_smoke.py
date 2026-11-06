# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from pathlib import Path

from openvad_train.export_smoke import main


def test_export_matches_committed_fixture(tmp_path):
    committed = Path(__file__).resolve().parents[3] / "inferencing/fixtures/smoke/linear.safetensors"
    main([str(tmp_path / "linear.safetensors")])
    assert (tmp_path / "linear.safetensors").read_bytes() == committed.read_bytes()
