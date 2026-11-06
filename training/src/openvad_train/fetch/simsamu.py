"""Simsamu: simulated French emergency calls (8 kHz, RTTM speaker turns) from Hugging Face, evaluation only."""

from openvad_train.fetch.common import download, hf_source, hf_tree, raw_dir, write_manifest


SIMSAMU_REPO = "medkit/simsamu"
SIMSAMU_REV = "6b639431f658d636c30dbf342074d723bf303e0f"


def fetch_simsamu() -> None:
    out = raw_dir("simsamu")
    files = [e for e in hf_tree(SIMSAMU_REPO, SIMSAMU_REV) if e["type"] == "file"]
    base = f"https://huggingface.co/datasets/{SIMSAMU_REPO}/resolve/{SIMSAMU_REV}"
    # The audio is in LFS and verified by SHA-256; the small text files are fixed by the pinned revision.
    for e in files:
        download(f"{base}/{e['path']}", out / e["path"], sha256=e.get("lfs", {}).get("oid"))
    info = {
        "dataset": "simsamu",
        "licence": "MIT (audio and annotations)",
        "use": "evaluation only, 8 kHz condition",
        "revision": SIMSAMU_REV,
        "sources": [hf_source(base, e) for e in files],
        "recordings": sum(1 for e in files if e["path"].endswith(".m4a")),
    }
    write_manifest(out, info)
