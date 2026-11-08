# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Run one dataset stage: `python -m openvad_train.dataset <stage> [args]`."""

import argparse
from collections.abc import Callable

from openvad_train.dataset import align, construct, handcheck, index, plan, publish, verify


# Each stage registers here with the task that adds it; a stage without an entry is not implemented yet.
STAGES: dict[str, Callable[[list[str]], None]] = {
    "index": index.main,
    "align": align.main,
    "handcheck-sample": handcheck.main_sample,
    "handcheck-score": handcheck.main_score,
    "plan": plan.main,
    "construct": construct.main,
    "publish": publish.main,
    "verify": verify.main,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="passed to the stage")
    ns = parser.parse_args(argv)
    if ns.stage not in STAGES:
        raise SystemExit(
            f"unknown or not yet implemented stage {ns.stage!r}; implemented: {', '.join(STAGES) or 'none'}"
        )
    STAGES[ns.stage](ns.args)


if __name__ == "__main__":
    main()
