"""CallHome telephone calls in all five languages from the gated TalkBank mirror on Hugging Face."""

from openvad_train.fetch import common
from openvad_train.fetch.common import download, hf_source, raw_dir, write_manifest


CALLHOME_REPO = "talkbank/callhome"
CALLHOME_REV = "17c8a153215aa7c50b805078fd6284ba81c2fc47"
CALLHOME_LANGS = ("eng", "deu", "jpn", "spa", "zho")


def require_hf_token() -> str:
    """The token from `HF_TOKEN` or `hf auth login`; the dataset is gated."""
    token = common.hf_token()
    if not token:
        raise RuntimeError(
            f"CallHome is gated: accept its terms at https://huggingface.co/datasets/{CALLHOME_REPO}, "
            "then run `hf auth login` or set HF_TOKEN"
        )
    return token


def callhome_files(tree: list[dict]) -> list[dict]:
    """The parquet shards of the languages we use."""
    return [
        e
        for e in tree
        if e["type"] == "file" and e["path"].split("/", 1)[0] in CALLHOME_LANGS and e["path"].endswith(".parquet")
    ]


def fetch_callhome() -> None:
    out = raw_dir("callhome")
    headers = {"Authorization": f"Bearer {require_hf_token()}"}
    files = callhome_files(common.hf_tree(CALLHOME_REPO, CALLHOME_REV, headers=headers))
    base = f"https://huggingface.co/datasets/{CALLHOME_REPO}/resolve/{CALLHOME_REV}"
    for e in files:
        download(f"{base}/{e['path']}", out / e["path"], sha256=e["lfs"]["oid"], headers=headers)
    info = {
        "dataset": "callhome",
        "licence": "CC BY-NC-SA 4.0 (HF mirror); TalkBank CC BY-NC-SA 3.0",
        "use": "train, dev, test (ADR 0023): real telephone conversations, 8 kHz originals at 16 kHz "
        "in the mirror; the labels are speaker turns, so pauses inside a turn count as speech",
        "revision": CALLHOME_REV,
        "languages": list(CALLHOME_LANGS),
        "sources": [hf_source(base, e) for e in files],
    }
    write_manifest(out, info)
