"""LibriVAD medium, generated locally by the upstream scripts at a pinned revision."""

import subprocess
import zipfile
from pathlib import Path

from openvad_train.fetch.common import (
    download,
    hf_source,
    hf_tree,
    is_complete,
    raw_dir,
    read_manifest,
    write_manifest,
)


LIBRIVAD_REPO = "LibriVAD/LibriVAD"
LIBRIVAD_REV = "bc641946f7df3a9515eb87d4ea194374f03a2dd3"
# Upstream converts only these two LibriSpeech splits to wav (Scripts/create_LibriSpeech_wav.py),
# so the unmodified generator writes no val split from dev-clean at this revision.
LIBRIVAD_SPLITS = ("train-clean-100", "test-clean")
LIBRIVAD_ZIPS = ("Files/Noises.zip", "Files/Forced_alignments.zip")


def librivad_files(tree: list[dict]) -> list[str]:
    """Upstream files to fetch: no evaluation models, and the zips are handled apart."""
    paths = (e["path"] for e in tree if e["type"] == "file")
    return sorted(
        p for p in paths if not p.startswith("Eval-ViT-MFCC/") and not (p.startswith("Files/") and p.endswith(".zip"))
    )


def librivad_uv_cmd(*script_args: str) -> list[str]:
    """Run an upstream script in its own environment, so its pins never touch ours."""
    return [
        "uv",
        "run",
        "--no-project",
        "--python",
        "3.12",
        "--with-requirements",
        "requirements.txt",
        "python",
        *script_args,
    ]


def librivad_counts(generated: Path) -> dict[str, int]:
    """Count the generated wav files per `<dataset>/<split folder>`."""
    splits = sorted(p for p in generated.glob("*/*") if p.is_dir())
    return {str(p.relative_to(generated)): sum(1 for _ in p.rglob("*.wav")) for p in splits}


def fetch_librivad() -> None:
    out = raw_dir("librivad")
    repo = out / "repo"
    base = f"https://huggingface.co/datasets/{LIBRIVAD_REPO}/resolve/{LIBRIVAD_REV}"
    tree = hf_tree(LIBRIVAD_REPO, LIBRIVAD_REV)
    for path in librivad_files(tree):
        download(f"{base}/{path}", repo / path)
    oids = {e["path"]: e["lfs"]["oid"] for e in tree if e["path"] in LIBRIVAD_ZIPS}
    by_path = {e["path"]: e for e in tree}
    sources = [hf_source(base, by_path[p]) for p in (*librivad_files(tree), *LIBRIVAD_ZIPS)]
    files = repo / "Files"
    extracted = {z: files / Path(z).stem for z in LIBRIVAD_ZIPS}
    # Labels/ appears early in create_labels.py, so only our own marker proves a finished setup.
    setup_done = out / ".setup-done"
    if not setup_done.exists():
        # setup.py skips a populated folder and deletes its zip even after a failed extraction,
        # so extract the zips ourselves and mark each one done; setup.py then skips them.
        for zip_path in LIBRIVAD_ZIPS:
            marker = out / f".{Path(zip_path).stem}-extracted"
            archive = repo / zip_path
            if marker.exists():
                archive.unlink(missing_ok=True)  # left behind by a crash after the marker
                continue
            download(f"{base}/{zip_path}", archive, sha256=oids[zip_path])
            with zipfile.ZipFile(archive) as z:
                z.extractall(extracted[zip_path])
            marker.touch()
            archive.unlink()
        subprocess.run(librivad_uv_cmd("setup.py", "EU"), cwd=repo, check=True)
        setup_done.touch()
    labels = sorted(p for p in files.glob("**/Labels") if p.is_dir())
    if not labels:
        raise RuntimeError(f"LibriVAD: no Labels directory under {files} after setup.py")
    generated = out / "generated"
    counts = _generate(repo, out, generated, "medium")
    info = {
        "dataset": "librivad",
        "licence": "CC BY-NC-SA 4.0",
        "use": "the train split is a pool source, reaching training only through OpenVAD Large (ADR 0023, ADR 0024); "
        "the test split is a benchmark, never trained on (ADR 0020)",
        "revision": LIBRIVAD_REV,
        "sources": sources,
        "size": "medium",
        "splits": list(LIBRIVAD_SPLITS),
        "labels_dir": labels[0].relative_to(out).as_posix(),
        "counts": counts,
    }
    write_manifest(out, info)
    # The markers guard a long setup and generation, so they stay until the manifest exists.
    for stem in (*(Path(z).stem + "-extracted" for z in LIBRIVAD_ZIPS), "setup-done", "generate-done"):
        (out / f".{stem}").unlink(missing_ok=True)


def _generate(repo: Path, out: Path, generated: Path, size: str) -> dict[str, int]:
    """Run the upstream generator once and check that every expected split has wav files."""
    # Upstream cannot resume, so a run without this marker regenerates the whole set.
    generate_done = out / ".generate-done"
    if not generate_done.exists():
        cmd = librivad_uv_cmd("create_LibriVAD.py", size, "--save_dir", str(generated))
        subprocess.run(cmd, cwd=repo, check=True)
        generate_done.touch()
    counts = librivad_counts(generated)
    expected = [f"{d}/{s}" for d in ("LibriSpeech", "LibriSpeechConcat") for s in LIBRIVAD_SPLITS]
    bad = [k for k in expected if counts.get(k, 0) <= 0]
    if bad:
        raise RuntimeError(
            f"LibriVAD: no wav files generated for {', '.join(bad)} under {generated}; "
            f"delete {generate_done} to regenerate"
        )
    return counts


def fetch_librivad_small() -> None:
    """LibriVAD `small`, the size the LibriVAD paper reports on, generated from the `librivad` setup."""
    medium = raw_dir("librivad")
    manifest = read_manifest(medium)
    if manifest is None or not is_complete(medium):
        raise RuntimeError(f"LibriVAD small reuses the setup in {medium}; run `just fetch librivad` first")
    out = raw_dir("librivad-small")
    out.mkdir(parents=True, exist_ok=True)
    counts = _generate(medium / "repo", out, out / "generated", "small")
    info = {
        "dataset": "librivad-small",
        "licence": "CC BY-NC-SA 4.0",
        "use": "the train split is for screening, run apart from OpenVAD Small (ADR 0023); the test split is "
        "the paper-comparison row of the LibriVAD benchmark, never trained on (ADR 0020)",
        "revision": LIBRIVAD_REV,
        # Generated from the medium setup's download, so there is none of its own: the pinned upstream revision.
        "sources": [{"url": f"https://huggingface.co/datasets/{LIBRIVAD_REPO}/tree/{LIBRIVAD_REV}"}],
        "size": "small",
        "splits": list(LIBRIVAD_SPLITS),
        # The labels belong to the clean source files, so the medium setup's labels cover small too.
        "labels_dir": f"../librivad/{manifest['labels_dir']}",
        "counts": counts,
    }
    write_manifest(out, info)
    (out / ".generate-done").unlink(missing_ok=True)
