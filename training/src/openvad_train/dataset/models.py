# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""`mfa_models.tsv`: the one list of languages the MFA sources are indexed and aligned in, with their models.

`status` is `downloaded` (used in a real run), `listed` (name and version are in the mfa-models release index but the
model was not downloaded) or `open` (ADR 0025 leaves the choice open and nothing runs).
"""

import csv
import dataclasses
from pathlib import Path


MODELS_TSV = Path(__file__).with_name("mfa_models.tsv")


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    language: str
    acoustic: str
    acoustic_version: str
    dictionary: str
    dictionary_version: str
    g2p: str | None
    g2p_version: str | None
    status: str
    note: str


def load_models(path: Path = MODELS_TSV) -> dict[str, ModelSpec]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    models = {}
    for r in rows:
        if None in r.values():  # a short row, such as one whose empty last cell an editor trimmed
            raise ValueError(f"{path.name}: row {r['language']!r} has fewer cells than the header")
        spec = ModelSpec(**{**r, "g2p": r["g2p"] or None, "g2p_version": r["g2p_version"] or None})
        if spec.status != "open" and not (spec.acoustic and spec.acoustic_version and spec.dictionary):
            raise ValueError(f"{path.name}: {spec.language} needs an acoustic model, a version and a dictionary")
        if bool(spec.g2p) != bool(spec.g2p_version):
            raise ValueError(f"{path.name}: {spec.language} has a G2P model without a version or the reverse")
        models[spec.language] = spec
    return models


def spec_for(models: dict[str, ModelSpec], language: str) -> ModelSpec:
    spec = models.get(language)
    if spec is None:
        raise ValueError(f"no row for language {language!r} in {MODELS_TSV.name}; have {sorted(models)}")
    if spec.status == "open":
        raise ValueError(f"{language}: no MFA model is chosen yet ({spec.note})")
    return spec
