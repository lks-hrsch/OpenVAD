import pytest

from fetch_support import clean_run, tar_bytes
from openvad_train.fetch import common, librispeech


def test_librispeech_rerun_skips_extracted_splits_and_checks_each_md5(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    md5s = {f"{s}.tar.gz": str(i) * 32 for i, s in enumerate(librispeech.LIBRISPEECH_SPLITS)}
    monkeypatch.setattr(librispeech, "read_text", lambda _: "".join(f"{m}  {n}\n" for n, m in md5s.items()))
    downloaded = []
    fail_on = {"test-clean.tar.gz"}

    def fake_download(url, dest, *, md5):
        downloaded.append((dest.name, md5))
        if dest.name in fail_on:
            raise ConnectionError("boom")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(tar_bytes(dest.name.removesuffix(".tar.gz")))
        return dest

    monkeypatch.setattr(librispeech, "download", fake_download)
    with pytest.raises(ConnectionError):
        librispeech.fetch_librispeech()
    assert downloaded == list(md5s.items())
    out = common.raw_dir("librispeech")
    assert (out / ".train-clean-100-extracted").exists()
    downloaded.clear()
    fail_on.clear()
    (common.cache_dir() / "train-clean-100.tar.gz").unlink()
    librispeech.fetch_librispeech()
    assert downloaded == [("test-clean.tar.gz", md5s["test-clean.tar.gz"])]
    assert (out / "LibriSpeech/train-clean-100/f.txt").exists()
    assert (out / "LibriSpeech/test-clean/f.txt").exists()
    clean_run("librispeech")
