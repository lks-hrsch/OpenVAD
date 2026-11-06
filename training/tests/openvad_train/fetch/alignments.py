import json
import zipfile

from fetch_support import clean_run
from openvad_train.fetch import alignments, common


def test_fetch_alignments_extracts_only_the_used_splits(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENVAD_DATA_DIR", str(tmp_path))
    entry = {"key": alignments.ALIGNMENTS_FILE, "checksum": "md5:" + "1" * 32, "links": {"self": "https://x/a.zip"}}
    other = {"key": "other.zip", "checksum": "md5:" + "2" * 32, "links": {"self": "https://x/o.zip"}}
    monkeypatch.setattr(alignments, "read_text", lambda _: json.dumps({"files": [other, entry]}))
    used = ["dev-clean/84/e.txt", "test-clean/61/f.txt", "train-clean-100/19/a.txt", "train-clean-360/20/g.txt"]
    fetched = []

    def fake_download(url, dest, *, md5=None):
        fetched.append((url, md5))
        dest.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(dest, "w") as z:
            for name in (*used, "train-other-500/20/b.txt", "test-other/21/c.txt", "dev-other/22/d.txt"):
                z.writestr(name, b"x")
        return dest

    monkeypatch.setattr(alignments, "download", fake_download)
    alignments.fetch_alignments()
    out = common.raw_dir("librispeech-alignments")
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.txt")) == used
    assert fetched == [("https://x/a.zip", "1" * 32)]
    assert clean_run("librispeech-alignments")["sources"] == [{"url": "https://x/a.zip", "md5": "1" * 32}]
