# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""FLEURS (Hugging Face `google/fleurs`) at a pinned revision, the languages with a pretrained MFA model."""

import tarfile

from openvad_train.fetch.common import (
    cache_dir,
    download,
    hf_source,
    hf_token,
    hf_tree,
    raw_dir,
    tar_filter,
    write_manifest,
)


FLEURS_REPO = "google/fleurs"
FLEURS_REV = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
# FLEURS languages with an MFA acoustic model (`*_mfa`, `*_cv` or VoxCommunis); not yet checked model by model.
FLEURS_LANGS = (
    "be_by", "bg_bg", "bn_in", "bs_ba", "ca_es", "cmn_hans_cn", "cs_cz", "cy_gb", "de_de", "el_gr", "en_us",
    "es_419", "et_ee", "fi_fi", "fr_fr", "ha_ng", "hr_hr", "hu_hu", "hy_am", "it_it", "ja_jp", "ka_ge", "kk_kz",
    "ko_kr", "ky_kg", "lg_ug", "mn_mn", "nl_nl", "pl_pl", "pt_br", "ro_ro", "ru_ru", "sr_rs", "sv_se", "sw_ke",
    "ta_in", "th_th", "tr_tr", "uk_ua", "uz_uz", "vi_vn", "yue_hant_hk",
)  # fmt: skip
FLEURS_SPLITS = ("train", "dev", "test")


def fleurs_files(tree: list[dict], lang: str) -> list[dict]:
    """The split TSVs and audio archives of one language directory; all six must be listed."""
    names = {f"{s}.tsv" for s in FLEURS_SPLITS} | {f"audio/{s}.tar.gz" for s in FLEURS_SPLITS}
    files = [e for e in tree if e["type"] == "file" and e["path"].split("/", 2)[2] in names]
    if len(files) != len(names):
        raise RuntimeError(f"FLEURS {lang}: the listing has {len(files)} of the {len(names)} expected files")
    return files


def fleurs_hours(tsv: str) -> tuple[int, float]:
    """Utterances and hours of one split; column 6 is the sample count at 16 kHz."""
    rows = [line.split("\t") for line in tsv.splitlines() if line]
    return len(rows), round(sum(int(r[5]) for r in rows) / 16000 / 3600, 2)


def fetch_fleurs() -> None:
    out = raw_dir("fleurs")
    out.mkdir(parents=True, exist_ok=True)
    token = hf_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    base = f"https://huggingface.co/datasets/{FLEURS_REPO}/resolve/{FLEURS_REV}"
    languages = {}
    sources = []
    markers = []
    for lang in FLEURS_LANGS:
        files = fleurs_files(hf_tree(FLEURS_REPO, FLEURS_REV, f"data/{lang}", headers), lang)
        sources += [hf_source(base, e) for e in files]
        for e in files:
            sha256 = e.get("lfs", {}).get("oid")
            if not e["path"].endswith(".tar.gz"):
                download(f"{base}/{e['path']}", out / e["path"].removeprefix("data/"), sha256=sha256, headers=headers)
                continue
            split = e["path"].rsplit("/", 1)[1].removesuffix(".tar.gz")
            markers.append(marker := out / f".{lang}-{split}-extracted")
            archive = cache_dir() / f"fleurs-{lang}-{split}.tar.gz"
            if marker.exists():
                archive.unlink(missing_ok=True)  # left behind by a crash after the marker
                continue
            download(f"{base}/{e['path']}", archive, sha256=sha256, headers=headers)
            with tarfile.open(archive) as tar:
                tar.extractall(out / lang / "audio", filter=tar_filter)
            marker.touch()
            archive.unlink()
        languages[lang] = {
            s: dict(zip(("utterances", "hours"), fleurs_hours((out / lang / f"{s}.tsv").read_text()), strict=True))
            for s in FLEURS_SPLITS
        }
    info = {
        "dataset": "fleurs",
        "licence": "CC BY 4.0",
        "use": "train (augmented and telephone channel, ADR 0023), dev, test",
        "revision": FLEURS_REV,
        "sources": sources,
        "languages": languages,
    }
    write_manifest(out, info)
    for marker in markers:
        marker.unlink(missing_ok=True)
