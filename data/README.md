# Data

Nothing in this directory is committed, except this file.

Code finds the data root through `OPENVAD_DATA_DIR` (default `./data`), so it can also point to a NAS mount.

## Fetching

`just fetch <dataset>` with `<dataset>` one of `librispeech`, `alignments`, `ava`, `librivad`, `demand`, `voices`, `callhome`, `swc`, `musan`, `librivad-small`, `picovoice`, `common-voice`, `simsamu`, `wham`, `mls`, `fleurs`, `voxpopuli`, `ami`, `speech-commands`.
Each run writes into `$OPENVAD_DATA_DIR/datasets/raw/<dataset>/` and ends with a `manifest.json` with the keys `dataset`, `licence`, `use`, `sources` (URL and message-digest algorithm 5 or SHA-256 per download), `files`, `bytes` and `retrieved`, plus fetcher-specific ones.
A re-run skips a dataset whose manifest verifies (file count and bytes match) and stops with an error if the manifest exists but does not verify (delete `manifest.json` to refetch); `librivad` regenerates the whole set unless its generation finished (marker `.generate-done`), because the upstream generator cannot resume. A fetch holds a lock file `cache/<name>.fetching` in the data dir (and a local lock under `~/.cache/openvad`) so two machines on one share cannot fetch the same dataset; it is removed when the run ends. A run killed on another machine leaves the file behind: delete it by hand once no fetch is running there. The same machine recovers a stale lock on its own.

- `ava` needs `ffmpeg` and `ffprobe` on the PATH (in the Nix dev shell; otherwise install them yourself). It checks the label and file lists against pinned SHA-256 hashes.
- `librivad` runs the pinned upstream scripts with `uv`, downloads about 61 GB for setup and writes about 210 GB.
  It takes hours.
- `callhome` takes all five languages of the mirror (`eng`, `deu`, `jpn`, `spa`, `zho`, about 11.2 GB). It is gated: accept its terms at <https://huggingface.co/datasets/talkbank/callhome>, then run `hf auth login` (or set `HF_TOKEN`).
- `voices` downloads the devkit only (about 29.5 GB, 19,200 wavs) over 8 parallel range connections, because its mirror caps each connection.
  The mirror publishes no checksums, so the size, the gzip CRC and the end of the archive are checked.
  The full release (about 448.8 GB) only re-records the same 3,903 utterances 256 times, so it is not fetched.
- `swc` downloads the whole 37.6 GB release (checked against its message-digest algorithm 5 checksum) and extracts English, German and Dutch; the manifest records each article's audio licence.
- `musan` extracts every file; the manifest records each file's licence from the committed table `training/src/openvad_train/fetch/musan_licences.tsv` (public domain, CC BY, CC BY-SA, CC BY-ND or no statement) and fails if a file has no entry.
- `librivad-small` runs the upstream generator at size `small` from the `librivad` setup (run `librivad` first); it is the size the LibriVAD paper reports on: its train split is used for screening, its test split as the paper-comparison row of the LibriVAD benchmark ([ADR 0023](../docs/project/adr/0023-dataset-composition.md)).
- `picovoice` downloads the Picovoice voice-activity-benchmark at a pinned commit and runs its own mixer on our `test-clean` and DEMAND (fetch `librispeech` and `demand` first): one 16 kHz wav (0 dB) and one label per 512-sample frame (0 silence, 1 unknown, 2 voice).
  The mixer holds the whole mix in memory (about 10 GB).
- `common-voice` needs a Mozilla Data Collective account: accept the terms on each dataset page (the 53 locales in `CV_LOCALES` of `fetch/commonvoice.py`), create an API key and put `MDC_API_KEY=...` in `.env` (git-ignored; `just` loads it). Who accepted which terms is in [ADR 0024](../docs/project/adr/0024-training-data-licence-policy.md).
  The page itself shows no terms: click the arrow next to Download and select "API / Python Access". If the terms are not yet accepted, this opens a four-checkbox agreement (CC0-1.0, no speaker identification, no re-hosting or re-sharing); tick all four and submit with the dialog's "API / Python Access" button, which needs no download. If they are accepted, the same menu item opens the API snippet directly.
  It downloads Scripted Speech 27.0 through the official API over parallel range connections (about 670 GB for the 53 locales with an MFA model). MDC allows 7 dataset downloads per organisation and day, so the download link is saved next to the archive and reused while it is valid, and when the limit is hit the run waits for the reset (midnight UTC plus 10 minutes) and continues with the same locale, which takes several days for all 53. It checks each archive against the SHA-256 published in `common-voice/cv-dataset`, extracts the metadata and the validated clips only and deletes each archive and its saved link after extraction (millions of small files, which takes days on the SMB share).
  The manifest records the official splits with clips and speakers, and the train pool (validated clips of speakers in neither dev nor test) with its hours, the ten largest speakers and the hours left under the 30 h per-speaker cap; the generator applies the cap, not the fetcher.
- `simsamu` downloads the Hugging Face repo at a pinned revision (57 MB); the audio is verified by SHA-256.
- `wham` downloads the WHAM! noise zip (18.2 GB, 18,194,930,748 bytes, message-digest algorithm 5 checksum published on <http://wham.whisper.ai/>, which answers over plain HTTP only; the Google Storage URL `whisper-public/wham_noise.zip` returns 404) over 8 parallel range connections and unpacks it (37.6 GB, 28,000 32-bit float stereo wavs at 16 kHz); the manifest records the audio format and the files and hours of `tr`, `cv` and `tt`.
- `mls` downloads Multilingual LibriSpeech as FLAC for German, French, Spanish, Italian, Dutch, Polish and Portuguese (about 365 GB, message-digest algorithm 5 checksums from the publisher) over 8 range connections, one language at a time, extracting and deleting each archive before the next.
- `fleurs` downloads the Hugging Face repo `google/fleurs` at a pinned revision for the 42 languages with an MFA model (about 100 GB, SHA-256 checked), extracts each audio archive and records utterances and hours per split.
- `voxpopuli` downloads the transcribed-data annotation tables for 13 languages and the 12 yearly session-audio tars (77 GB, no checksums published, so the size and the end of each tar are checked, which finds an early end and a hole at a member boundary but not one inside a member's data); segments are cut by the generator.
- `ami` downloads the manual annotations from Edinburgh and the OpenSLR 16 `headset.tar.gz` (24.4 GB, the same files as Edinburgh, which answers 403 to per-file requests from Python; no checksum published, so the size, the gzip CRC and the end of the archive are checked) over 8 range connections, then extracts only the individual headsets of the full-corpus-ASR partition (about 46 GB) and deletes the archive; headsets absent from the archive are listed as `missing`.
- `speech-commands` downloads Speech Commands v0.02 and its test set (2.5 GB, SHA-256 from TensorFlow Datasets) and records the official splits; the test split is `testing_list.txt` (11,005 clips), the separate 4,890-clip test set is not used.
- Fearless Steps has no fetcher: the corpus needs registration and acceptance of the challenge terms ([datasets note](../docs/notes/datasets.md#per-source-facts)).
- TalkBank's CallHome transcripts (`https://talkbank.org/data/ca/CallHome/<lang>?f=zip`) need a TalkBank login; no fetcher takes them.
- `demand` takes only the 16 kHz zips; SCAFE exists only at 48 kHz and is listed in the manifest as `missing_16k`.
- Our splits ([ADR 0023](../docs/project/adr/0023-dataset-composition.md)): official dev and test where a source has them, otherwise about 10% each by speaker, reader or call; at most 30 h per speaker in each dataset (applied by the generator). LibriSpeech is not trained on directly (LibriVAD covers `train-clean-100`).
- `dev-clean` and `test-clean` are never read directly: LibriVAD builds its test split from `test-clean` and its validation split from `dev-clean`, and they reach validation and test through LibriVAD only.
- LibriVAD at the pinned revision has no validation split: upstream converts only `train-clean-100` and `test-clean` to wav, so `generated/` holds those two; the manifest lists them as `splits`. `just fetch librivad-val` (not written yet) is to generate the `medium` val split from `dev-clean` into `datasets/raw/librivad-val/` with its own manifest, so the unmodified train and test build stays as it is.
- LibriVAD `medium` takes every 10th utterance of `train-clean-100` with 54 mixtures each, so about 10 h of unique speech appear 54 times by design.
- `just fetch rirs` (not written yet) is to fetch the room impulse responses of BUT ReverbDB and OpenSLR 28 ([ADR 0021](../docs/project/adr/0021-telephone-channel-simulation.md)).

## ffmpeg for dataset construction

The dataset builder runs ffmpeg as a tool for the stored codec legs and checks the encoders it needs at start-up (`preflight`).
The Nix dev shell provides `ffmpeg-full`, which has them all.
Elsewhere, install an ffmpeg with `libopencore_amrnb` (AMR-NB), `libvo_amrwbenc` (AMR-WB), `libopus` and the built-in `g722`, for example configured with `--enable-version3 --enable-libopencore-amrnb --enable-libvo-amrwbenc --enable-libopus`.
`ffmpeg -encoders | grep -E "amr|opus|g722"` lists them.
ffmpeg stays a separate process, so none of its GPL or version-3 code is linked into the release.
`dataset/codec_delays.json` records the ffmpeg version its delays were measured with.

## Layout

```text
$OPENVAD_DATA_DIR/
  datasets/raw/
    librispeech/              LibriSpeech/<split>/<speaker>/<chapter>/
    librispeech-alignments/
    ava-speech/
      ava_speech_labels_v1.csv
      audio/<video_id>.flac
      missing.txt
    librivad/
      repo/                   pinned upstream snapshot, setup outputs under Files/
      generated/
    demand/                   <scene>/ch01.wav ... ch16.wav
    voices/                   VOiCES_devkit/
    callhome/                 eng/, deu/, jpn/, spa/, zho/   parquet shards
    swc/                      english/, german/, dutch/   <article>/audio.ogg, aligned.swc, audiometa.txt, ...
    musan/                    musan/{music,noise,speech}/<source>/*.wav, LICENSE, ANNOTATIONS
    librivad-small/           generated/ (labels in librivad/)
    picovoice/                benchmark_speech.wav, benchmark_labels.txt, repo/
    common-voice/             <locale>/ (53 locales)   <release>/<locale>/{train,dev,test,validated,...}.tsv, clips/*.mp3
    simsamu/                  <recording>/<recording>.{m4a,rttm,srt,json}, metadata.csv
    wham/                     wham_noise/{tr,cv,tt}/*.wav, wham_noise/metadata/
    mls/                      mls_<language>/{train,dev,test}/{transcripts.txt,audio/<speaker>/<book>/*.flac}
    fleurs/                   <lang>/{train,dev,test}.tsv, <lang>/audio/<split>/*.wav
    voxpopuli/                annotations/asr_<lang>.tsv.gz, raw_audios/   session .ogg files
    ami/                      annotations/, audio/<meeting>/<meeting>.Headset-<n>.wav
    speech-commands/          speech_commands_v0.02/<word>/*.wav, speech_commands_test_set_v0.02/
  datasets/processed/         later
  models/checkpoints
  cache/                      partial downloads, deletable
```

LibriVAD frame labels are produced by its `setup.py` somewhere under `repo/Files/`; the manifest records the directory as `labels_dir`.
AVA-Speech audio starts at video time 900 s, so a label time maps to audio time as `t_audio = t_label - 900`.

## Datasets and licences

Licences are recorded, not filtered on ([ADR 0024](../docs/project/adr/0024-training-data-licence-policy.md)); the Use column says which of our splits a source feeds.
Benchmark data and every dev and test pool never enter training.
Every training dataset and anything rendered from them stay on the owner's network share; ADR 0024 says what else may hold them.

| Dataset | Licence | Use |
| --- | --- | --- |
| LibriSpeech (`train-clean-100`, `dev-clean`, `test-clean`) | CC BY 4.0 | source of LibriVAD and Picovoice only; `dev-clean` and `test-clean` reach validation and test through LibriVAD, never directly, and `test-clean` is also benchmark audio (ADR 0020, ADR 0023) |
| LibriSpeech alignments (Zenodo 2619474) | CC BY 4.0 | labels for VOiCES (ADR 0023) and the `test-clean` benchmark only; keeps the `train-clean-360` alignments for VOiCES speakers |
| AVA-Speech | labels CC BY 4.0; audio belongs to the YouTube uploaders, never redistributed | benchmark |
| LibriVAD `medium` | CC BY-NC-SA 4.0 | train split: pool source as generated (NonConcat and Concat), reaching training only through OpenVAD Large; val split (from `dev-clean`, generated by `librivad-val`): validation only; test split: test and benchmark (ADR 0020); val and test are never trained on (ADR 0023, ADR 0024) |
| DEMAND (Zenodo 1227121, 16 kHz) | CC BY-SA 3.0 assumed (record metadata says CC BY 4.0) | noise: channels 1 to 12 train, 13 and 14 dev, 15 and 16 test (ADR 0023) |
| VOiCES devkit | CC BY 4.0 | train, dev, test, unmodified (ADR 0023): the 12 `dev-clean` speakers are dropped, the 64 `train-clean-100` speakers are train only, the rest split 80/10/10 by speaker |
| SWC English, German and Dutch (Hamburg FDR record 1875) | CC BY-SA 4.0 corpus and alignments; audio per article (mostly CC BY-SA, some public domain, CC BY, GFDL-only or unknown), recorded in the manifest | train, dev, test by reader (augmented, ADR 0023) |
| MUSAN (OpenSLR 17) | per file, recorded in the manifest: public domain, CC BY, CC BY-SA; 32 music files CC BY-ND 3.0, 14 without a licence statement | train, dev, test (noise only, one shared pool; music is not used) |
| LibriVAD `small` | CC BY-NC-SA 4.0 | test split: paper-comparison row of the LibriVAD benchmark (ADR 0020), never trained on; train split: screening, run apart from OpenVAD Small (ADR 0023) |
| Picovoice voice-activity-benchmark | Apache-2.0 code; LibriSpeech `test-clean` (CC BY 4.0) mixed with DEMAND | benchmark |
| CallHome `eng`, `deu`, `jpn`, `spa`, `zho` (HF `talkbank/callhome`) | CC BY-NC-SA 4.0 (TalkBank: CC BY-NC-SA 3.0) | train, dev, test by call, unmodified at 16 kHz with 8 kHz content, ±0.25 s masked around each turn boundary (ADR 0023) |
| Common Voice Scripted Speech 27.0, 53 locales (Mozilla Data Collective) | CC0-1.0; the platform terms forbid re-hosting or re-sharing the audio, also modified, and any attempt to identify speakers | train (validated clips of speakers outside dev and test, augmented), dev, test (official) |
| Simsamu (HF `medkit/simsamu`) | MIT, audio and annotations | benchmark (8 kHz condition, see ADR 0020) |
| WHAM! noise (`tr`, `cv`, `tt`) | CC BY-NC 4.0 | noise (City): `tr` train, `cv` dev, `tt` test only (ADR 0023) |
| BUT ReverbDB (fetcher to come) | CC BY 4.0, about 8.7 GB | room impulse responses, rooms split 80/10/10 by hash (ADR 0021) |
| OpenSLR 28, RIRS_NOISES (fetcher to come) | Apache-2.0, about 1.3 GB | room impulse responses, as BUT ReverbDB (ADR 0021) |
| MLS (DE, FR, ES, IT, NL, PL, PT; FLAC) | CC BY 4.0 | train (augmented), dev, test (official) |
| FLEURS (42 languages, HF `google/fleurs`) | CC BY 4.0 | train (augmented), dev, test (official) |
| VoxPopuli transcribed (13 languages) | CC0 data; code CC BY-NC 4.0, not used | train, dev, test by speaker, unmodified; rows without a speaker ID train only |
| AMI Meeting Corpus, individual headsets | CC BY 4.0 (Edinburgh release); OpenSLR 16 states CC BY-NC-SA 2.0 (modified) | train, dev, test (full-corpus-ASR), unmodified |
| Speech Commands v0.02 | CC BY 4.0 | train, dev, test (`testing_list.txt`), unmodified |

Datasets and third-party weights are never redistributed; the scripts fetch them.
See [ADR 0018](../docs/project/adr/0018-datasets.md), [ADR 0024](../docs/project/adr/0024-training-data-licence-policy.md) and the [datasets note](../docs/notes/datasets.md).
