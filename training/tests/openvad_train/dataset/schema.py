# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

from openvad_train.dataset.schema import rank, sample_seed


def test_rank_is_stable():
    assert rank("swc/a") == rank("swc/a")
    assert rank("swc/a") != rank("swc/b")


def test_rank_is_64_bit():
    assert 0 <= rank("swc/a") < 2**64


def test_seed_depends_on_version_and_id_only():
    assert sample_seed(1, "v1", "x") == sample_seed(1, "v1", "x")
    assert sample_seed(1, "v2", "x") != sample_seed(1, "v1", "x")
    assert sample_seed(2, "v1", "x") != sample_seed(1, "v1", "x")
    assert sample_seed(1, "v1", "y") != sample_seed(1, "v1", "x")


def test_seed_parts_do_not_run_together():
    assert sample_seed(1, "v1", "x") != sample_seed(1, "v", "1x")
