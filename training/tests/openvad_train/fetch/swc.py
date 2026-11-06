# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import zipfile

import pytest

from fetch_support import clean_run, tar_with
from openvad_train.fetch import common, swc


@pytest.mark.parametrize(
    ("audiometa", "expected"),
    [
        ("{{int:license-header}}\n{{self|cc-by-sa-3.0}}", "CC BY-SA"),
        ("{{self|gfdl|cc-by-sa-3.0,2.5,2.0,1.0}}", "CC BY-SA"),
        ("{{GFDL|migration = relicense}}", "CC BY-SA"),
        ("{{gfdl|migration=redundant}}", "CC BY-SA"),
        ("{{gfdl|migration=not-eligible}}", "GFDL"),
        ("{{gfdl-en}}", "GFDL"),
        ("{{self|cc-by-3.0}}", "CC BY"),
        ("{{PD-user|someone}}", "PD/CC0"),
        ("{{self|cc-zero}}", "PD/CC0"),
        ("{{int:license-header}}", "unknown"),
        ("{{self|cc-by-sa-3.0}}{{Information|Description=Spoken article, pdf version}}", "CC BY-SA"),
        ("{{self|cc-by-sa-3.0}}\n{{en|Spoken by Foo, PD}}", "CC BY-SA"),
        ("{{self|cc-by-sa-3.0|author=PD Fan}}", "CC BY-SA"),
        ("{{cc-by-nc-sa-3.0}}", "unknown"),
        ("{{cc-by-nd-3.0}}", "unknown"),
        ("{{pdf}}", "unknown"),
        # Forms found in the real audiometa files.
        (" {{ CC-BY-SA 3.0}}", "CC BY-SA"),
        ("{{pd user|Rfc1394}}", "PD/CC0"),
        ("{{PD-user-w|en|wikipedia|Tdxiang}}", "PD/CC0"),
        ("{{GFDL-user|APPER|de|migration=relicense}}", "CC BY-SA"),
        ("{{cc-by-sa-all}}", "CC BY-SA"),
    ],
)
def test_swc_audio_licence(audiometa, expected):
    assert swc.audio_licence(audiometa) == expected


def _swc_zip(dest):
    with zipfile.ZipFile(dest, "w") as z:
        for lang in swc.SWC_LANGS:
            tar_with(dest.parent / "inner.tar", f"{lang}/A/audiometa.txt", b"{{cc-by-4.0}}")
            z.write(dest.parent / "inner.tar", f"SWC_{lang}.tar")
    (dest.parent / "inner.tar").unlink()


def test_fetch_swc_extracts_each_language_once_and_records_licences(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    fetched = []

    def fake_download_ranges(url, dest, *, md5=None):
        fetched.append((url, md5))
        dest.parent.mkdir(parents=True, exist_ok=True)
        _swc_zip(dest)
        return dest

    monkeypatch.setattr(swc, "download_ranges", fake_download_ranges)
    out = common.raw_dir("swc")
    # A crash after the English marker: English is on disk and must not be extracted again.
    (out / "English/Old").mkdir(parents=True)
    (out / "English/Old/audiometa.txt").write_text("{{cc-by-sa-4.0}}")
    (out / ".English-extracted").touch()
    swc.fetch_swc()
    assert not (out / "English/A").exists()
    manifest = clean_run("swc")
    assert fetched == [(swc.SWC_URL, swc.SWC_MD5)]
    assert manifest["audio_licences"] == {"English/Old": "CC BY-SA", "German/A": "CC BY", "Dutch/A": "CC BY"}
