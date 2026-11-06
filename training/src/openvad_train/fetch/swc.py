"""Spoken Wikipedia Corpora 2.0, English, German and Dutch, from the Hamburg research data repository."""

import re
import tarfile
import zipfile
from pathlib import Path

from openvad_train.fetch.common import cache_dir, download_ranges, raw_dir, tar_filter, write_manifest


SWC_URL = "https://www.fdr.uni-hamburg.de/record/1875/files/swc-2.0.zip"
SWC_MD5 = "b6e8567e39cc0b44bc03b05e576fee69"
SWC_LANGS = ("English", "German", "Dutch")
# Most permissive first; each article takes the most permissive licence its audio is offered under.
SWC_LICENCE_RANK = ("PD/CC0", "CC BY", "CC BY-SA", "GFDL", "unknown")


def licence_options(template: str) -> set[str]:
    """Licences one Wikimedia Commons licence template offers, e.g. `self|gfdl|cc-by-sa-3.0`."""
    t = template.lower().replace(" ", "")
    # The 2009 licence migration relicensed GFDL files marked `relicense` (or `redundant`) under CC BY-SA 3.0.
    migrated = "migration=relicense" in t or "migration=redundant" in t
    name, *args = t.split("|")
    # Only a template's name and the arguments of `self` name a licence; other arguments are free text.
    items = [name] if name != "self" else [i for arg in args if "=" not in arg for i in arg.split(",")]
    found = set()
    for item in items:
        if item.startswith(("cc-by-nc", "cc-by-nd")):
            continue
        if item.startswith(("cc-zero", "cc0", "pd-", "pduser")) or item == "pd":
            found.add("PD/CC0")
        elif item.startswith("cc-by-sa"):
            found.add("CC BY-SA")
        elif item.startswith("cc-by"):
            found.add("CC BY")
        elif item.startswith("gfdl"):
            found.add("CC BY-SA" if migrated else "GFDL")
    return found


def audio_licence(audiometa: str) -> str:
    """The most permissive licence offered by the `{{...}}` templates of an article's audio description."""
    found = set().union(*(licence_options(t) for t in re.findall(r"\{\{([^{}]*)\}\}", audiometa)))
    return min(found, key=SWC_LICENCE_RANK.index) if found else "unknown"


def swc_licences(out: Path) -> dict[str, str]:
    """Audio licence per article, keyed by `<language>/<article>`."""
    return {
        meta.parent.relative_to(out).as_posix(): audio_licence(meta.read_text(errors="replace"))
        for meta in sorted(out.glob("*/*/audiometa.txt"))
    }


def fetch_swc() -> None:
    out = raw_dir("swc")
    out.mkdir(parents=True, exist_ok=True)
    archive = cache_dir() / "swc-2.0.zip"
    markers = [out / f".{lang}-extracted" for lang in SWC_LANGS]
    with zipfile.ZipFile(download_ranges(SWC_URL, archive, md5=SWC_MD5)) as z:
        for lang, marker in zip(SWC_LANGS, markers, strict=True):
            if marker.exists():
                continue
            with z.open(f"SWC_{lang}.tar") as member, tarfile.open(fileobj=member, mode="r|") as tar:
                tar.extractall(out, filter=tar_filter)
            marker.touch()
    licences = swc_licences(out)
    info = {
        "dataset": "swc",
        "licence": "CC BY-SA 4.0 (corpus and alignments); audio per article, see audio_licences",
        "use": "train (augmented and telephone channel, ADR 0023), dev, test by reader; every article, any licence",
        "sources": [{"url": SWC_URL, "md5": SWC_MD5}],
        "languages": list(SWC_LANGS),
        "audio_licences": licences,
    }
    write_manifest(out, info)
    for path in (archive, *markers):
        path.unlink(missing_ok=True)
