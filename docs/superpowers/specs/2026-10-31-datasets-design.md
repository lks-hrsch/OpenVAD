Historical: superseded by ADRs 0023 and 0024 and the current fetchers; kept as the record of the 2026-10-31 design, aligned on 2026-11-06 with the ADRs, the notes and the fetchers, which win wherever they differ.

# Dataset retrieval: LibriSpeech, AVA-Speech, LibriVAD

Date: 2026-10-31. Branch: `feat/datasets`.

## Intent

Fetch the public datasets onto the NAS with one reproducible command each, so training (RQ1) and evaluation (RQ3) can start.
Every fetch records where the data came from and how to check it, because the thesis claims full reproducibility on public data.

The first design had two model variants by data licence, and commit b3f2ef8 added ADR 0019 for them, later removed.
That option was not taken: ADR 0024 decision 4 is one model, no licence variants, and the number 0019 stays unused.

## Goals

- `just fetch <dataset>` fetches one dataset into `$OPENVAD_DATA_DIR/datasets/raw/<dataset>/` and writes a `manifest.json`.
- Re-running is safe: a dataset whose manifest exists and verifies is skipped; an interrupted run resumes or restarts cleanly. A manifest that does not verify stops the run with an error; delete it to refetch.
- The design also asked for a one-off NAS read-speed check after LibriSpeech; no result is recorded in the repo.

## Non-goals (deferred, YAGNI)

- Resampling to 16 or 8 kHz (needs the OpenVAD resampler, which does not exist yet).
- Frame labels on the 10 ms grid, processed shards, data loaders.
- LibriSpeech splits other than `train-clean-100`, `dev-clean` and `test-clean`.
- The 9 AVA-Speech videos missing from the mirror (no YouTube download).
- Shards off the NAS. ADR 0024 decision 6 lets the workstation hold temporary local copies; any other copy needs a new decision by the owner.

## Anti-goals

- Committing any data, weights or third-party dataset code to the repo.
- Vendoring LibriVAD scripts (CC BY-NC-SA) into the repo.
- Resampling or downmixing AVA audio during fetch (would bypass the one resampler).
- Training on `dev-clean` or `test-clean` (LibriVAD builds its test split from `test-clean` and is meant to build validation from `dev-clean`; using them leaks the evaluation).

## Constraints

- Data root: `OPENVAD_DATA_DIR` (default `./data`); the owner sets it to `/Volumes/datasets/OpenVAD`, an SMB share on the Mac.
- Works without Nix: Python via uv, ffmpeg as an external binary (in the flake dev shell for Nix users; documented for others).
- Fetch code is stdlib-only Python in the `openvad_train` package; no new dependency.
- English, British spelling, no em dashes, Conventional Commits.

## Datasets

| Dataset | Licence | Role | Source | Size |
| --- | --- | --- | --- | --- |
| LibriSpeech `train-clean-100`, `dev-clean`, `test-clean` | CC BY 4.0 | source of LibriVAD and Picovoice only; `test-clean` is benchmark audio | OpenSLR SLR12 tarballs | 7.3 GB on disk |
| LibriSpeech alignments (Lugosch, Montreal Forced Aligner) | CC BY 4.0 | labels for VOiCES and the `test-clean` benchmark only | Zenodo record 2619474, `librispeech_alignments.zip` | 623 MB zip, 1.7 GB on disk |
| AVA-Speech | labels CC BY 4.0; audio belongs to the YouTube uploaders, never redistributed | independent real-world benchmark | labels from Google, audio from the AVA S3 mirror | 151 clips × 15 min |
| LibriVAD `medium` | CC BY-NC-SA 4.0 | test split: benchmark; train split: pool source, reaching training only through OpenVAD Large | HF `LibriVAD/LibriVAD`, generated locally | about 210 GB output, about 61 GB setup |

### LibriSpeech

- URL: `https://openslr.elda.org/resources/12/<split>.tar.gz` (EU mirror).
- Verify each archive against `https://www.openslr.org/resources/12/md5sum.txt` (e.g. `train-clean-100.tar.gz` = `2a93770f6d5c6c964bc36631d331a522`).
- Download to `cache/`, verify, extract into `datasets/raw/librispeech/` (the archive yields `LibriSpeech/<split>/<speaker>/<chapter>/`), then delete the archive.

### LibriSpeech alignments

- URL: `https://zenodo.org/records/2619474/files/librispeech_alignments.zip`; verify against the MD5 that the Zenodo API reports for the file.
- Extract the four splits in `ALIGNMENT_SPLITS` into `datasets/raw/librispeech-alignments/`.
- LibriVAD ships what looks like the same alignments, but inside its NC repo; VOiCES and the `test-clean` benchmark take their labels only from Zenodo.

### AVA-Speech

- Labels: `https://research.google.com/ava/download/ava_speech_labels_v1.csv` (columns `video_id, start_s, end_s, label`; classes `NO_SPEECH`, `CLEAN_SPEECH`, `SPEECH_WITH_MUSIC`, `SPEECH_WITH_NOISE`; 160 videos; labels cover 900 s to 1800 s of each video; timestamps are absolute within the video).
- File names: `https://s3.amazonaws.com/ava-dataset/annotations/ava_file_names_trainval_v2.1.txt` maps a video ID to `<id>.<mp4|mkv|webm>`; 151 of the 160 IDs are present. Both lists are stored and checked against pinned SHA-256 hashes.
- Audio: `ffmpeg -nostdin -loglevel error -y -rw_timeout <timeout> -ss 900 -i https://s3.amazonaws.com/ava-dataset/trainval/<file> -t 900 -vn -c:a flac audio/<id>.flac`.
  Use `-t 900`, not `-to 1800`: input seeking resets output timestamps to 0, so `-to 1800` would cut 30 minutes.
  ffmpeg reads only the byte ranges it needs (30 s took 8 s in a test); output keeps the native rate and channels (the test file: AAC, 44.1 kHz, stereo).
- Store the label CSV as downloaded. The extracted audio starts at video time 900 s; the manifest records this offset so labels map with `t_audio = t_label - 900`.
- If ffprobe reports a non-zero container `start_time`, record it per file in the manifest.
- IDs not on the mirror go to `missing.txt`.

### LibriVAD

Run the published generator unmodified (approach A below).

1. Download the repo snapshot at a pinned revision (current `main`: `bc641946f7df3a9515eb87d4ea194374f03a2dd3`) into `datasets/raw/librivad/repo/`, using `https://huggingface.co/datasets/LibriVAD/LibriVAD/resolve/<rev>/<path>` per file from the HF tree API. Skip `Eval-ViT-MFCC/` (evaluation code and models, not needed).
2. Download `Files/Noises.zip` and `Files/Forced_alignments.zip` at the same revision, check them against the SHA-256 of the HF tree, extract them into `repo/Files/` and delete them.
   `setup.py` skips a populated folder, so the pinned revision is what gets used.
3. In `repo/`: `uv run --no-project --python 3.12 --with-requirements requirements.txt python setup.py EU`.
   This downloads its own `train-clean-100`, `dev-clean`, `test-clean`, converts only `train-clean-100` and `test-clean` to wav, builds LibriSpeechConcat and labels (`.npy`, 1 = speech, per sample at 16 kHz).
4. In `repo/`: `uv run ... python create_LibriVAD.py medium --save_dir ../generated`.
   This writes `generated/<LibriSpeech|LibriSpeechConcat>/<split>/<noise>/<snr>/*.wav` for train and test (no val split at the pinned revision), 9 noises, SNRs −5 to 20 dB in 5 dB steps.
5. Write the manifest: revision, sources, and wav counts per dataset and split.

Generation takes hours; it runs from the Mac and writes over SMB.

## Layout

The layout is in `data/README.md`.

Each `datasets/raw/<dataset>/manifest.json` holds the keys `dataset`, `licence`, `use`, `sources` (URL and checksum per download), `files`, `bytes` and `retrieved`, plus fetcher-specific ones; for AVA also per-file sha256, the 900 s offset and `start_time` values.

## Approaches considered

### LibriVAD generation

- A. Run `setup.py` and `create_LibriVAD.py` unmodified; they download their own copy of three LibriSpeech splits (~7 GB). **Picked by the owner.** Byte-identical to the published reference build, and their `shutil.rmtree(flac_dir)` never sees our raw LibriSpeech.
- B. Point the generator at our LibriSpeech via a patch or symlinks. Saves 7 GB but changes the reference build, and the `rmtree` risks deleting our raw data. Rejected.

### LibriSpeech source

- OpenSLR tarballs with published MD5s (picked: canonical, checkable, no extra dependency).
- HF `openslr/librispeech_asr` parquet: same audio repackaged, needs `datasets`; no gain.

### AVA audio

- ffmpeg seek over HTTP, audio only, labelled 15 minutes (picked: ~40 h of audio instead of ~140 GB of video).
- Download full videos, then cut: wasteful, and video is not wanted.

## Code

- `training/src/openvad_train/fetch/`: `common.py` (paths, downloads, manifests), one module per dataset and `__main__.py`: `python -m openvad_train.fetch <dataset>`, with the 19 names in `FETCHERS`.
  Reads `OPENVAD_DATA_DIR`, prints what it does, fails loudly on a checksum mismatch or a missing `ffmpeg`, holds a lock per dataset, and never deletes anything under `datasets/raw` except its own partial output and markers.
- `justfile`: `fetch dataset` recipe.
- `flake.nix`: add `pkgs.ffmpeg` to the dev shell.
- Tests mirror the tree in `training/tests/openvad_train/fetch/`, no network.

## Docs

- `data/README.md`: the layout and the licence and use per dataset.
- ADR 0018 (stays `Open`): the sources, their roles, reasons and scale.

## NAS speed check

Planned as a one-off command after LibriSpeech, to decide whether training needs a local copy or shards. No result is recorded in the repo; ADR 0024 decision 6 now allows temporary local copies on the workstation.

## Builder's calls (decided in this document, not by the owner)

- EU OpenSLR mirror; archives deleted after verified extraction.
- Manifest granularity: archive checksums plus counts; per-file sha256 only for AVA.
- Skip `Eval-ViT-MFCC/` from the LibriVAD snapshot.
- ffmpeg as an external binary, not PyAV.

## Left to the builder

- Concurrency of AVA extraction (a few parallel ffmpeg processes).
- Exact manifest field names.
- Where `setup.py` writes its labels inside `repo/Files/`; record it in the manifest and `data/README.md`.

## Extension (from 2026-11-01)

The owner added further datasets; licences and uses are in ADR 0018 and `data/README.md`.

- `demand`: DEMAND 16 kHz zips from Zenodo 1227121, MD5-checked.
- `voices`: the VOiCES devkit from the public S3 bucket; the full release is not fetched.
- `callhome`: CallHome `eng`, `deu`, `jpn`, `spa` and `zho` parquet shards from the gated `talkbank/callhome` at a pinned revision, SHA-256-checked; NC.
- `common-voice`: Common Voice Scripted Speech 27.0, the 53 locales of `CV_LOCALES`, through the Mozilla Data Collective API.
- `simsamu`: evaluation only.
- `swc`, `musan`, `librivad-small`, `picovoice`, `wham`, `mls`, `fleurs`, `voxpopuli`, `ami` and `speech-commands`: see `data/README.md`.
- A `librilight` fetcher was added in `dec90e8` and removed in `3e2e365`.

`download_ranges` in `fetch/common.py` (8 connections, 64 MiB pieces, resumable by a piece log) serves mirrors that cap each connection at 0.55 to 1.3 MB/s, which made the full VOiCES release a ten-day download; `ami`, `common-voice`, `mls`, `musan`, `swc`, `voices`, `voxpopuli` and `wham` use it.
The datasets are fetched one after another, smallest first, because the SMB share times out under concurrent small-file writes.
