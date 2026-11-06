import hashlib

import pytest

from fetch_support import clean_run, hf_server
from openvad_train.fetch import common, simsamu


def test_simsamu_verifies_lfs_audio_and_keeps_the_repo_layout(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    audio, oid = b"audio", hashlib.sha256(b"audio").hexdigest()
    tree = [
        {"type": "directory", "path": "rec"},
        {"type": "file", "path": "rec/rec.m4a", "lfs": {"oid": oid}},
        {"type": "file", "path": "rec/rec.rttm"},
    ]
    listing = f"https://huggingface.co/api/datasets/medkit/simsamu/tree/{simsamu.SIMSAMU_REV}?recursive=true"
    base = f"https://huggingface.co/datasets/medkit/simsamu/resolve/{simsamu.SIMSAMU_REV}/rec"
    files = {f"{base}/rec.m4a": b"corrupt", f"{base}/rec.rttm": b"turns"}
    hf_server(monkeypatch, {listing: tree}, files)
    out = common.raw_dir("simsamu")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        simsamu.fetch_simsamu()
    assert common.read_manifest(out) is None

    files[f"{base}/rec.m4a"] = audio
    simsamu.fetch_simsamu()
    manifest = clean_run("simsamu")
    assert (out / "rec/rec.m4a").read_bytes() == audio
    assert (out / "rec/rec.rttm").read_bytes() == b"turns"
    assert manifest["recordings"] == 1
    assert manifest["sources"] == [{"url": f"{base}/rec.m4a", "sha256": oid}, {"url": f"{base}/rec.rttm"}]
