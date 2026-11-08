# Dataset construction spec

Status: draft for owner review, 2026-11-06.

This spec says how `openvad_train` turns `$OPENVAD_DATA_DIR/datasets/raw` into the OpenVAD dataset: forced-alignment labels, constructed audio samples, and a Parquet index mapping each sample to its split and tier.

Items marked (owner, 2026-11-06) were decided in the design questionnaire.
Unmarked values are proposals the owner may correct; the ones that matter most are listed under [Proposals to check](#proposals-to-check).
Where this spec differs from an ADR, the ADR is rewritten in the same branch ([ADR changes](#adr-changes)).

## Why

OpenVAD's claim is reproducibility on public data (ADR 0004), and RQ1 needs one training set built by open code.
The fetchers exist, but nothing yet builds a sample, so training cannot start.
Most speech has transcripts without timings, so forced alignment is part of this work.

## Goals

- Plan every clip of the pool once (owner, 2026-11-06), about 46,000 h of train audio.
- Produce one Parquet row per sample with a pointer to its 16 kHz audio, labels on the 10 ms grid and metadata (owner, 2026-11-06).
- Point unmodified sources at their files in `datasets/raw`, and store only constructed audio in `data/` (owner, 2026-11-06).
- Provide one Parquet index mapping sample to train, validation or test, and to Small or Large (owner, 2026-11-06).
- Read and write all Parquet asynchronously with Polars (owner, 2026-11-06).
- Load metadata and pointers locally with Hugging Face `datasets`, with `load_audio(row)` reading the audio, as in Granary-v2's manifests-only design.
- Rebuild any sample from its row and the raw data.
- Stop cleanly when the share is full and resume once the owner adds storage (owner, 2026-11-06).
- A build stopped early is an interim set: it may be used for training and screening, but headline results wait for the final frozen set (owner, 2026-11-06).
- New storage extends the same mount; all shard paths are relative to the dataset root (owner, 2026-11-06).

## Non-goals

- Uploading anything. "Hugging Face compliant" is a local format; ADR 0024 forbids redistribution.
- Test-grid, Test-unseen-noise and benchmark preparation. They are later configurations of this generator.
- The full 500-clip Praat check of ADR 0025. Only its 100-clip pilot is in scope, see [Quality bar](#quality-bar). The word tables keep what the later check's sampling needs.
- The training loop, model and DataLoader. Only the narrowband transform they call is in scope.
- Packet loss, PLC, DTX and CNG (owner, 2026-11-06, for packet loss). See [Deferred](#deferred).
- Several mixtures per clip: at 2 mixtures the set needs at least 6 TB.

## Anti-goals

- A sample whose labels disagree with its audio timing.
- A second resampler for 8 and 16 kHz conversion (ADR 0009).
- Any step that changes a sample's content because another source was added or a budget changed.
- Rendered data or training data leaving the share and the workstation's temporary copies (ADR 0024).

## Constraints

- Python 3.12 in the uv workspace, no build logic only in the flake; `tools/` builds use standalone tooling.
- The share has 3.1 TB free today (`df`, 2026-11-06) and about 3.0 TB after the Common Voice fetch ends (about 11 November).
- The estimated full set is about 2.97 TB of constructed audio with an unmeasured FLAC ratio, so the first full build may stop early until storage is added.
- Millions of small files on SMB take days (`data/README.md`), so loose files per sample are not an option.
- Common Voice is still being fetched; plans are frozen only after it ends. Development uses the 20 extracted locales.
- A plain ffmpeg lacks `libopencore_amrnb` and `libvo_amrwbenc`; the builder needs a build with them (the Nix shell's `ffmpeg-full` has both, [data/README.md](../../../data/README.md#ffmpeg-for-dataset-construction)), and construction fails early without it.

## Approaches

The decisive choice was storage, with three alternatives.

- Parquet with embedded audio: the native Hugging Face form, but unmodified sources would be copied, and every split file carries all its audio.
- Loose FLAC files like LibriVAD: easiest to browse, and Hugging Face `Audio` decodes them, but about 15 M files on SMB.
- **Split Parquet files with pointers (owner, 2026-11-06):** constructed audio in tar shards under `data/`, unmodified audio left in raw. Thousands of files, about 180 GB saved, and the same manifests-only idea as Granary-v2. Hugging Face returns pointers rather than decoded audio.

For the channel chain:

- Everything precomputed (ADR 0021 as it stood): G.711 and the 16 to 8 kHz step are frozen per sample.
- Whole chain at train time: more variety, but ffmpeg gives about 90 to 190x real time per core against roughly 460x needed.
- **Stored codecs with narrowband at train time (owner, 2026-11-06):** cheap random narrowband in training, and a filter change does not force a rebuild of wideband samples.

For alignment with LibriVAD, the owner chose its fixed SNR grid and noise categories, keeping ADR 0023's padding, clean and competing talker conditions, and one mixture per clip because of disk size.

## Output

### Layout

```text
$OPENVAD_DATA_DIR/datasets/processed/openvad-v1/
  README.md          Hugging Face YAML: splits pointing at the three Parquet files
  manifest.json      versions, build flags, seeds, licences per source
  index.parquet      one row per sample: split and tier
  train.parquet      pointers, labels and metadata
  validation.parquet
  test.parquet
  data/              constructed audio only
    train/00000.tar  about 1 GB of uncompressed FLAC members, in rank order
    validation/00000.tar
    test/00000.tar
```

Working state sits next to it, under `datasets/processed/work/<name>/` with the build configuration's name, so a slice run never overwrites the full build's state: `index/<source>.parquet` and `index/manifest.json` (every stage writes its manifest into its own directory), `words/<source>/<language>/<batch>.parquet`, `plan/plan.parquet`, `fragments/manifest.json`, and one `fragments/<shard>.parquet` and `fragments/<shard>.failures.jsonl` per finished shard.

- `publish` merges the fragments and as-is rows into the split files.
- Polars reads with `scan_parquet(...).collect_async()` and writes with `sink_parquet(..., lazy=True)` under `collect_all_async`, inside `asyncio`, through one small module. Polars marks `collect_async` unstable, so the version is pinned in `uv.lock`.
- CPU work stays in worker processes; async covers Parquet I/O.

### Pointers

- `audio_filepath` is relative to `$OPENVAD_DATA_DIR`, so a mount change needs no rewrite, and new storage extends the same mount.
- Unmodified sources (`treatment = as_is`) point into `datasets/raw/...`, and their raw files are never copied or changed (owner, 2026-11-06). That covers VoxPopuli, AMI, CallHome, Speech Commands, VOiCES and LibriVAD.
- Constructed samples (`treatment = augmented`) point at a tar in `data/` with `byte_offset` and `byte_length`.
- `load_audio(row) -> np.ndarray` in `dataset/audio.py` is the one reader. `audio_format` selects `flac_tar`, `file` (with `offset` and `duration`, for WAV and Ogg) or `callhome_parquet` (with `row_in_file`, because the mirror embeds audio in Parquet).
- `verify` checks that every pointer resolves, and `manifest.json` records each raw source's manifest hash. Changing raw data invalidates the set.
- Seeking inside long VoxPopuli session files may be slow during training; milestone 4 measures it and reports to the checkpoint.

### Tiers and Hugging Face

- The Small size is not decided yet (owner, 2026-11-06).
- Train samples carry a stable random order, `rank = blake2(sample_id)`, so Small can later be a rank cut-off over everything except LibriVAD. This sets `in_small` in `index.parquet` and `train.parquet` only, no audio is rebuilt.
- Small is therefore a strict subset of Large and leaves out LibriVAD `medium` (ADR 0017).
- Validation and test are the same rows for both tiers (ADR 0023).
- The README defines one config, with `train`, `validation` and `test` naming the three Parquet files. Small is `ds.filter(lambda r: r["in_small"])`.
- `load_dataset` returns pointers, labels and metadata. It never decodes audio; that follows Granary-v2's manifests-only design.

### LibriVAD splits

- Train: `medium` train, as generated, NonConcat and Concat, Large only.
- Validation: `medium` val from LibriSpeech `dev-clean`, about 27.4 h NonConcat and 35.2 h Concat (owner, 2026-11-06).
- Test: `medium` test from `test-clean`, as generated (owner, 2026-11-06). It stays ADR 0020's benchmark, reported per dataset, and is never trained on.
- `dev-clean` and `test-clean` reach validation and test through LibriVAD only, and never reach train.
- Val does not exist yet: `create_LibriSpeech_wav.py` converts only two splits at the pinned revision, while `create_LibriVAD.py` and `create_LibriSpeechConcat.py` already handle `val`, and `Files/Noises/*_val.wav` and the `dev-clean` alignments exist.
- `just fetch librivad-val` generates it into `datasets/raw/librivad-val/`, with its own manifest, so the unmodified train and test build stays as it is.

### Columns of the split Parquet files

| Column | Type | Meaning |
| --- | --- | --- |
| `sample_id` | string | stable, `<source>/<clip or chunk id>` |
| `audio_filepath` | string | relative to `$OPENVAD_DATA_DIR` |
| `audio_format` | string | `flac_tar`, `file` or `callhome_parquet` |
| `byte_offset`, `byte_length` | int64, nullable | constructed samples, position inside the tar |
| `offset`, `duration` | float64 | seconds; `offset` is 0.0 for constructed samples |
| `row_in_file` | int64, nullable | CallHome only |
| `labels` | list of uint8 | one per 10 ms frame: 0 non-speech, 1 speech, 2 masked |
| `band` | string | `wide`, `narrow_leg` or `narrow_native`, see [train time](#train-time-narrowband) |
| `in_small` | boolean, nullable | train only, null until Small is set |

Metadata columns follow: `rank`, `source`, `language`, `speaker` (nullable), `licence`, `treatment` (`augmented`, `as_is`), `speech_ratio`, `condition`, `noise_source_ids`, `noise_offsets`, `snr_db` (nullable), `tir_db` (nullable), `baseline_snr_db` (nullable), `room`, `rir_id` (nullable), `handset`, `agc`, `clip_drive_db`, `suppression`, `codec`, `bitrate`, `codec_delay_ms`, `transients`, `gap_ms`, `dither_dbfs`, `segments`, `aligner` (nullable), `seed`.

These cover the manifest fields of ADR 0023; loss pattern is absent because v1 has no packet loss.

`plan.parquet` holds these columns, with null pointers on constructed rows until `construct`, plus `split`.
It also holds `clip_id`, `shard`, `source_path`, `source_offset_n`, `source_n`, `speech_offset_n`, `total_n`, `speech_ms` and `mask_ms`.
These are plan-only and dropped at publish, except what pointers need.

`index.parquet` holds `sample_id`, `split`, `in_small`, `source`, `treatment` and `audio_filepath`: the quick map of the whole set, without labels.

The `manifest.json` records `plan_version`, global seed, commit, ffmpeg version and build flags, MFA version and models, RIR and noise manifests, the delay table and the resampler coefficients' hash.

### File rate

- Everything stored is 16 kHz (owner, 2026-11-06). Sources above 16 kHz are converted with ffmpeg (one German Common Voice MP3 measured 44.1 kHz), with the version recorded (ADR 0009).
- Nothing is stored at 8 kHz, which replaces "stored at 8 kHz" in ADR 0021 and ADR 0023.
- CallHome's mirror holds 8 kHz telephone audio in 16 kHz files. It is stored unchanged, with `band = narrow_native`.
- Every pointed-at raw file is 16 kHz already (`data/README.md`, datasets note); `verify` checks this on a sample.

## Pipeline

Stages run in order, each writing its output and a manifest.
`just dataset <stage>` calls `python -m openvad_train.dataset <stage>`.

### 1. `index`: what clips exist

One adapter per source in `dataset/sources/<source>.py`, mirroring `fetch/`, returns clip rows: clip ID, audio path, offset and duration, language, speaker, split, licence, transcript or label source.

- Splits follow ADR 0023's table.
- AVA-Speech, Fearless Steps, Picovoice and Simsamu are never read. LibriSpeech `dev-clean` and `test-clean` are never read directly. LibriVAD val may enter validation only, and LibriVAD test may enter test only (ADR 0024).
- Reserved speakers: 5% of speakers per language, at least one, on train and on dev and test separately, feed babble and competing talkers only and are never foreground (ADR 0022, ADR 0023).
- FLEURS has no speaker IDs, so it reserves nothing. Languages with under 50 speakers draw babble from any language; Japanese and Mandarin keep ADR 0023's p = 0.5.

### 2. `align`: labels as intervals

The canonical artefact is a table per batch: clip ID, word or turn intervals in ms, mask intervals, `kind` (`word`, `turn`, `frame`), MFA metrics and provenance.
TextGrids stay on local temporary disk and are converted, then deleted.

Non-MFA sources:

| Source | Input | Notes |
| --- | --- | --- |
| SWC | word timings | an unaligned token inside a gap, or a gap of 2 s or more, makes that gap masked; a word over 5 s is masked (proposal 13) |
| AMI | manual word XML | vocal sounds excluded; crosstalk is non-speech |
| CallHome | turns | union of both sides is speech; ±0.25 s masked around every turn boundary of either side (ADR 0022) |
| VOiCES | Zenodo alignments | shifted by the delay measured by cross-correlation |
| LibriVAD | frame labels | used as they are: per-sample `int16` to 10 ms by majority |
| Speech Commands, background noise | none | non-speech |

MFA sources are Common Voice, MLS, FLEURS, VoxPopuli (`invalid` rows dropped) and Speech Commands.

- **Batches:** clips are taken in rank order per speaker, one MFA run per batch, keeping each speaker's clips together for fMLLR.
- **Thresholds:** per language and source, the drop thresholds that depend on data are fitted once on a calibration batch and then frozen, so later batches cannot change earlier decisions. The first batch is not used, because rank order puts the most prolific speaker first and one voice would set the thresholds. The calibration batch takes clips in rank order, at most 10 per speaker, up to 1000 clips (a clip without a speaker counts as a speaker); its alignments are discarded and its clips go through the normal batches. Only calibration clips that pass the fixed rules (wider beam, word and phone length, coverage, first 20 ms) are fitted on and counted, because the thresholds should describe acceptable alignments. With fewer than 50 of them the language is not aligned and the manifest records `too_few_to_calibrate`. Its composition, rank cut, the counts before and after the fixed rules and the fitted rules are in `rules.json` and the manifest.
- **Cap:** the 30 h cap walks rank order over surviving clips of each speaker; once a speaker reaches 30 h, that speaker's remaining clips are never aligned. This saves MFA time and keeps the cap correct after drops. `align` walks per source, language and speaker, which is an upper bound; the cap that counts is `apply_cap_rank` keyed (source, speaker) over the surviving MFA timings in rank order, applied in `plan`. Both stages count audio seconds (`duration_s`) for MFA sources, because `align` must decide before any alignment exists; only SWC counts speech seconds.
- **Drop rules:** ADR 0025 as written (owner, 2026-11-06), with Speech Commands exempt from the speaking-rate and first-20-ms rules. The speaking-rate rule scales the median absolute deviation by 1.4826 (the usual estimate of sigma; unscaled, three deviations are 2.02 sigma and drop 4.3% of a clean population), stored in `rules.json`; the owner is to confirm this reading.
- **Out-of-vocabulary words:** a G2P model where one exists, passed to `mfa align` as `--g2p_model_path` (`--use_g2p` exists only on `align_hf`, `align_one` and `adapt` in MFA 3.4.3); otherwise `spn` for Italian and Dutch (owner, 2026-11-06).
- **Text:** digits map to `<unk>`; Japanese and Mandarin use the MFA 3.4.3 tokenisers.
- **Models:** one committed table, `dataset/mfa_models.tsv`: language, acoustic model, dictionary, G2P, each with the release version to download, a note and a `status` (`downloaded`, `listed` or `open`), following `docs/notes/forced-alignment.md`. The provenance of every row records the version read from the downloaded model.
- **No path:** a language with an acoustic model but no lexicon gets a rule-based lexicon (Epitran or XPF). If it fails the quality bar, the language follows the failure rule below (owner, 2026-11-06).
- Audio is read from a temporary local copy, per batch, on the workstation (ADR 0024).

#### Quality bar

Before any language enters the full run (owner, 2026-11-06):

- **Pilot coverage:** a small sample of every language using `*_cv` or VoxCommunis models with over 200 h in the pool, plus English, German, Japanese and Mandarin.
- **Automatic bar per language:** alignment failure rate and the share of clips removed by the drop rules. The thresholds are set from the pilot distribution and recorded in the manifest.
- **Hand check:** the owner labels 100 clips in Praat, 25 each from English, German, Italian and Catalan, sampled across score quantiles and labelled blind to MFA. Reported as frame agreement on the 10 ms grid, and onsets and offsets within 20 and 50 ms.
- **Tooling:** `just dataset handcheck-sample` writes WAVs and empty TextGrids, and `just dataset handcheck-score` writes the report.
- **Failure rule:** a failing language is dropped and recorded automatically. If it has over 500 h in the pool, the owner decides instead.

### 3. `plan`: every draw, no audio

`plan.parquet` has one row per sample and resolves every random draw, using `seed = blake2(global_seed, plan_version, sample_id)` and never a running sequence, so adding a source does not change other samples.
Running it prints hours and bytes per source, split and condition.

### 4. `construct`: a pure function of the plan

`construct_sample(plan_row, raw_data) -> (flac_bytes, labels)`, for `augmented` rows only. As-is rows skip this stage.

- One worker per shard, 11 on the workstation, each writing a local uncompressed tar plus a fragment Parquet with offsets and labels, verifying them, copying to `<shard>.tmp` on the share, then `os.replace`.
- A shard is done when its tar, fragment and sidecar SHA-256 exist. The fragment is renamed into place last, so a fragment means a finished shard, and a tar without one is rebuilt.
- Before each shard starts, free space is checked against estimated shard size plus a 50 GB margin, and the run stops cleanly if short. A re-run continues.
- Train shards are started in rank order, so a stop for space leaves a clean prefix. A failed worker can leave a hole, because shards after it may finish; a re-run fills it.
- Labels are rasterised from the audio produced, never from file metadata, because real Common Voice MP3s without a Xing header overstate their length by 5.6 to 7.5 ms.
- A row fails if its source is damaged, if the decode is more than 160 samples (one 10 ms frame) from `source_n`, if its RMS over the word union at G = 0 is under −70 dBFS, or if its noise is digital silence under the words. A failing row is retried once, then left out and logged to `fragments/<shard>.failures.jsonl`.
- A source that cannot give one byte after that means the share is failing, not the clip, and aborts the run.
- A shard with no row built, or over 5% of its rows failed, aborts without publishing, so a re-run rebuilds it.
- `fragments/manifest.json` records the plan version, global seed, plan SHA-256 and ffmpeg version and flags. A re-run with any of them changed is refused, and so are shards with no manifest. A completed run adds the total `n_failed`, also logged as a warning.

### 5. `publish` and `verify`

`publish` merges fragments and as-is rows into `train.parquet`, `validation.parquet` and `test.parquet`, and writes `index.parquet`, `README.md` and `manifest.json`.
A build missing planned shards is marked `state: interim`.
`verify` checks: sidecar hashes, every pointer resolves, every constructed row decodes, as-is rows decode on a seeded sample of up to 200 per source (decoding every raw file over SMB would take weeks), label length against audio length on all constructed rows and on that sample, one index row per sample, speech ratio and condition shares per source and split, and speakers disjoint between splits where ADR 0023 says they can be.

### Labels

- All label logic runs on integer-millisecond intervals and rasterises last: `rasterise(speech_ms, mask_ms, n_samples, shift_ms) -> frames`, with `round(n_samples / 160)` frames so the label length matches the audio. G is applied earlier by `apply_gap_rule`. Seconds are converted with `round(s * 1000)`; floats are rejected.
- A frame is masked if any mask covers it; otherwise it takes the majority class.
- G = 200 ms (owner, 2026-11-06), applied only at rasterisation, so a change of G never reruns MFA.
- Audio decisions, such as the 20% coverage drop, padding to 33% and TIR over active frames, use the word union at G = 0, so a G ablation changes labels and leaves audio bytes identical.
- Shifts apply in order: padding offset, then codec delay. RIRs are peak-aligned to zero, so reverberation adds no shift.
- The reverb tail after speech offset is non-speech (owner, 2026-11-06).
- LibriVAD labels are used as they are, without the gap rule (proposal; ADR 0023 left it open).
- Overlapping CallHome turns count as speech (proposal; ADR 0023 left it open).

### Sample cutting

| Source | Cut |
| --- | --- |
| Common Voice, FLEURS | one clip, one sample, padded |
| MLS, SWC | split at word gaps longer than G into chunks of at most 6.6 s speech, each padded to 10 to 20 s |
| VoxPopuli | as annotated; above 30 s split at gaps |
| AMI, CallHome | cut at non-speech gaps to at most 30 s; not padded. A sample runs longer only where speech crosses the limit (a word, or overlapping turns in CallHome) or a last piece under 1 s joins the one before |
| Speech Commands | one clip, one sample |
| VOiCES | whole wav, every re-recording; distractor-only noise files not used |
| LibriVAD `medium` | every generated wav, one row each; each utterance appears 54 times by design |

- A stretch still over 6.6 s of speech has no gap longer than G, so it is split between two words where the budget would overflow, at that gap's midpoint with no margin. A last piece under 1 s is evened with the one before it. Only a single word over the limit stays whole.
- Such a cut can sit on a label edge, which is accepted because noise or dither covers the whole sample.
- A chunk also ends where its span would pass 20 s, so a total above 20 s comes only from a stretch that cannot be cut.
- A chunk with under 200 ms of speech yields no sample and is counted in the plan manifest.
- Cutting uses G = 200 ms whatever G the labels use, so a G ablation leaves audio bytes identical.

### Acoustic mix, augmented sources

Common Voice, MLS, FLEURS and SWC.
As-is sources skip this and the stored channel chain.

- **Padding:** to about 33% speech (owner, 2026-11-06, keeping ADR 0023). Common Voice's natural clip edges count towards it. Speech goes at a random offset, in one block.
- **Clean padding:** Gaussian dither at −90 to −70 dBFS, so the model cannot detect digital silence.

| Condition | Share | Source |
| --- | --- | --- |
| Clean | 15% | |
| Competing talker | 15% | ADR 0022: reserved speakers, 1 to 3 segments covering 10 to 50%, TIR 0 to 15 dB, each with an RIR and a random low-pass from 1 to 4 kHz |
| Babble | 70/9 % | 6 energy-normalised talkers summed, as LibriVAD; same language, reserved speakers |
| SSN | 70/9 % | Gaussian white noise through a 12th-order LPC all-pole filter fitted to one reserved speaker, as LibriVAD |
| City | 70/9 % | WHAM!, `tr` train, `cv` validation, `tt` test, one random channel per draw |
| Domestic, Nature, Office, Public, Street, Transport | 70/9 % each | DEMAND scenes by first letter (D, N, O, P, S, T), as LibriVAD, one random channel per draw; channels 1 to 12 train, 13 and 14 validation, 15 and 16 test (owner, 2026-11-06) |

- Categories are LibriVAD's 9 plus clean and competing talker (owner, 2026-11-06). Music is not used (owner, 2026-11-06).
- Noise comes from our pools, not LibriVAD's noise files (owner, 2026-11-06).
- MUSAN noise files are sorted into the 9 categories by keyword (owner, 2026-11-06), so the 6.2 h adds variety to the thin DEMAND categories. A committed table, `dataset/musan_noise_categories.tsv`, holds the result of a one-time keyword pass plus manual review; files that fit no category are listed there as unused. Its licences come from the existing `fetch/musan_licences.tsv`.
- MUSAN music is not used.
- **SNR:** uniform over {−5, 0, 5, 10, 15, 20} dB (owner, 2026-11-06, LibriVAD's grid), replacing ADR 0023's bimodal ranges.
- SNR is computed from speech RMS over speech-active frames and noise RMS over the same span, then applied to the noise over the whole sample.
- **Common Voice baseline:** WADA-SNR on each clip estimates its speech power, and `baseline_snr_db` is recorded. Brouhaha is deferred because it needs an isolated `numpy<2` environment.
- **Transients:** synthetic clicks and DTMF only (owner, 2026-11-06), p = 0.3, peak ratio 0 to −18 dB. Typing and handling noise are dropped from ADR 0023.

### Stored channel chain, augmented sources

| Step | p | Values |
| --- | --- | --- |
| Room | 0.5 | real RIRs: BUT ReverbDB (CC BY 4.0, 8.7 GB) and OpenSLR 28 (Apache-2.0, 1.3 GB) (owner, 2026-11-06); rooms split 80/10/10 by hash; peak aligned to zero |
| Handset | 0.7 | high-pass 100 to 300 Hz, low-pass 3.4 to 7 kHz, two peaks ±6 dB |
| AGC | 0.5 | target −26 to −16 dBFS, attack 5 to 20 ms, release 100 to 500 ms |
| Soft clipping | 0.15 | drive 3 to 12 dB |
| Noise suppression | 0.3 | Wiener or spectral subtraction with scipy |

Stored codec leg:

| Codec | Share | Bitrate | Path |
| --- | --- | --- | --- |
| None | 40% | | `band = wide` |
| G.722 | 30% | 64 kbit/s | ffmpeg, `band = wide` |
| AMR-WB | 15% | 6.6 to 23.85 kbit/s | ffmpeg, `band = wide` |
| Opus | 5% | 12 to 32 kbit/s | ffmpeg, `band = wide` |
| AMR-NB | 10% | 4.75 to 12.2 kbit/s | decimate, ffmpeg, interpolate, `band = narrow_leg` |

- Each codec's delay is measured once by cross-correlation and stored as a fixed table (ADR 0021). Labels use it.
- ffmpeg runs as a subprocess, about 50 to 110 ms per 10 s sample as measured for G.722 and Opus.
- EVS is left out until its licence is checked; ADR 0021 allows that.

### Train-time narrowband

`dataset/narrowband.py` provides `apply_narrowband(audio, band, draw) -> audio` (owner, 2026-11-06):

| `band` | Behaviour |
| --- | --- |
| `wide` | with probability `p_narrow`: decimate, G.711, interpolate |
| `narrow_leg` | always: decimate, G.711, interpolate, because the AMR-NB caller leg always reaches the network through G.711 |
| `narrow_native` | nothing |

- `p_narrow` is a training parameter, default 0.35, so the gateway log query no longer blocks construction.
- It applies to any `wide` sample, as-is sources included; `treatment` lets training exclude them.
- G.711 is A-law in 90% and μ-law in 10% of draws (ADR 0021).
- G.711 is a numpy lookup table in `training/`; the engine has no user for it, so it stays out of `openvad-core`.
- Delay: decimator and interpolator total 66 samples at 16 kHz (33 each, 4.125 ms, 0.4125 frames) at 67 taps, which rounds to 0 on the 10 ms grid. The function computes the shift from the coefficients, and a test asserts the result.
- Training draws randomly per sample. Validation and test use a seed derived from `sample_id`, so scoring is repeatable.

Scoring conditions over validation and test (owner, 2026-11-06):

| Condition | Processing | Purpose |
| --- | --- | --- |
| 16 kHz | stored audio | ADR 0020 |
| 8 kHz | decimate, interpolate; no G.711 | ADR 0020: bandwidth is the only variable |
| Telephone | decimate, G.711, interpolate | labelled third condition: the target domain |

The 8 kHz condition lags the 16 kHz one by the 4.125 ms of the resampler round trip, so bandwidth is the only variable up to that delay, which is under a frame and left uncompensated.

`narrow_leg` and `narrow_native` samples stay in every condition and are reported through `band`.
`narrowband.py` therefore exposes a resample-only path as well as the G.711 path.
The scorer itself is deferred (ADR 0020).

### Resampler

Fixed now (owner, 2026-11-06), which answers ADR 0009's open question.

- `openvad-core` gets a half-band decimator and interpolator: causal, linear-phase, 67 taps, at least 70 dB stopband, committed coefficients, explicit `mul_add` in canonical order, scalar tier only.
- `openvad-rs` exposes them on NumPy arrays through the `numpy` crate, because `Vec<f32>` from Python lists is too slow here.
- A committed script designs the coefficients, and the coefficient hash goes in `manifest.json`.
- Tests: frequency response against the stated stopband, group delay, known vectors, and a Python and Rust comparison at `1e-6`.
- 63 taps and 70 dB were starting values. Measured across 3500 to 4500 Hz: 63 taps reach only -69.6 dB, so the committed design has 67 taps, the smallest valid length (-73.3 dB, one-way group delay 33 samples at 16 kHz, DC gain 0.0019 dB above unity).

## Tools

### MFA in `tools/`

The owner wants the latest release, built from source from submodules (owner, 2026-11-06). The details, pins, build steps and licences are in `tools/README.md`.

- `tools/montreal-forced-aligner`: git submodule pinned to the tag of the latest release, v3.4.3 (2026-10-08). Never `main`, which is heading to 4.0 and removes the tokenisers.
- `tools/kaldi`: git submodule at a pinned commit, compiled as shared libraries into `tools/kaldi/dist` (the `KALDI_ROOT`) by `just build-mfa`.
- `tools/mfa` is a uv project of its own (fallback 1 below, taken because an editable path source on an unfetched submodule breaks `uv sync`). It holds the editable MFA, `kalpy-kaldi` built against `KALDI_ROOT`, and the packages MFA needs; `pynini` is built by the recipe with the conda-forge API patch that `kalpy-kaldi` needs. `just setup` and `uv sync --all-packages` work without the submodules, and `just mfa <args>` runs `mfa`.
- `initdb` and `pg_ctl` come from the system, from a PostgreSQL server with pgvector. They are a documented requirement.
- Target machine: the NixOS x86_64 workstation.

Facts from PyPI (2026-11-06): MFA is `py3-none-any`; `pynini` has a cp312 manylinux wheel; `kalpy-kaldi` is source only; `baumwelch` is not on PyPI, and the PyPI `ngram` looks like an unrelated package.
Both should be needed only for training models, not for aligning with pretrained ones. The first build on macOS aligned with `german_mfa` without either (`tools/README.md`); the pilot confirms this on the workstation.

Fallbacks, tried in order if a pilot gate fails:

1. A separate uv project in `tools/mfa/`, if the sdist breaks the workspace `uv lock` or `uv sync`. Taken, because of the sync.
2. conda-forge `kaldi` as `KALDI_ROOT` only, if compiling Kaldi fails. This needs a note to the owner, since it is a second tool.

### New dependencies

- `training`: `polars`, `soundfile`, `scipy`, `datasets`.
- `openvad-rs`: `numpy` crate.
- New fetchers: `just fetch rirs` for BUT ReverbDB and OpenSLR 28, and `just fetch librivad-val`, following the `fetch/` conventions.

## Delivery order

Each milestone ends in something runnable, and later ones build on it.

1. **Vertical slice:** SWC German, DEMAND, G.722, no MFA or resampler, none of the other chain steps or transients, a few hundred samples. Covers index, labels, plan, construct, publish, verify, and `load_dataset`. Measures FLAC ratio and throughput.
2. **Resampler and narrowband:** the filter, bindings, G.711, the train-time transform, and AMR-NB.
3. **MFA gate:** build it, then pilot on English, German, Italian, Catalan and Japanese. Measures throughput and settles the model table. Gates are listed below.
4. **All adapters:** remaining sources and the MFA batches.
5. **Full chain:** all conditions, RIRs, reserved speakers, transients and all stored codecs.
6. **Owner checkpoint:** the owner reviews measured FLAC ratio, MFA throughput, quality bar results, gateway log numbers if available, and every proposal and placeholder. No full-scale run starts without this approval (owner, 2026-11-06).
7. **Full run:** freeze the plan after Common Voice ends, align, construct until the share is full, and report.

### Gates

| Gate | Pass | If it fails |
| --- | --- | --- |
| FLAC ratio on milestone 1 | measured | new size estimate goes to the checkpoint |
| MFA builds from the submodules | `mfa align` runs on German with a pretrained model | fallbacks above |
| `*_cv` models on MFA 3.4.3 | pilot languages align | rule-based lexicon route, then the failure rule |
| Japanese and Mandarin tokenisers | clips align | failure rule |
| Quality bar | automatic bar and hand check reported | failure rule |
| MFA throughput, alignment alone | under 30 days for the full run | owner decides at the checkpoint whether to cut clips or wait |
| AMR through ffmpeg | round trip works with measured delay | drop AMR legs and report |

## Testing

Tests mirror the tree, with `training/tests/openvad_train/dataset/...`.

- Determinism: `construct_sample` twice gives identical bytes, and adding a source leaves other plan rows unchanged.
- Labels: gap rule, mask precedence, majority rule, padding and delay shifts on hand-made intervals.
- Audio and label alignment: an impulse at a known time survives padding and each codec at the expected frame, within one frame.
- Mix: measured SNR and TIR match the drawn value within 0.5 dB on synthetic signals.
- Shards: a killed worker leaves no partial shard visible, and a re-run skips finished shards.
- Pointers: `load_audio` returns the same samples as the tar member or raw file, for each `audio_format`, and a broken pointer fails `verify`.
- Index: every sample once, with `in_small` null until set; Small is a subset of Large once set and has no LibriVAD.
- Polars: the async read and write helpers round-trip `labels`, and a failed write leaves no visible file.
- Hugging Face: `load_dataset` on a small fixture gives the three splits with pointers and labels, and does not try to read the tars.
- Split guards: benchmark paths raise, and LibriVAD val and test raise outside validation and test.
- Resampler: see above.
- No real data in tests. Fixtures are generated; the parity fixtures rule still holds.

## ADR changes

Rewritten in this branch, each marked (owner, 2026-11-06), status unchanged:

- 0009: resampler filter fixed in this phase; train-time narrowband.
- 0018: add BUT ReverbDB and OpenSLR 28; MUSAN contributes noise only, sorted into categories.
- 0021: stored 16 kHz only; narrowband at train time; G.711 in `training/`; packet loss, DTX and CNG deferred; proposed probabilities recorded; EVS out for now.
- 0022: G = 200 ms; delay handling; reverb tail; CNG deferred.
- 0023: 9 categories plus clean and competing talker, SNR grid, shares, every clip, Small undecided as a rank cut-off, no 8 kHz storage, pointer layout, cutting rules, CallHome, drop rules, synthetic transients, WHAM! `tt` in test, DEMAND channels split 12/2/2 replacing the shared pool, LibriVAD val and test in the split files.
- 0024: LibriVAD val and test sit in validation and test; they still never train.
- 0025: drop rules confirmed, Speech Commands exemption, `spn` handling, MFA in `tools/`, rule-based lexicon fallback.
- 0020: Test-broad City noise; telephone as a third labelled condition; BG-FAR available only on competing-talker samples; the paper's unseen noises (Babble, SSN, Domestic) are seen by OpenVAD, which is stated with the LibriVAD results.
- 0017: Small and LibriVAD val follow the new layout.

`docs/notes/datasets.md` follows, including a correction: LibriVAD `medium` takes every 10th utterance with 54 mixtures each, so it holds about 10 h of unique speech rather than 101 h.
`docs/notes/forced-alignment.md`, `data/README.md` and `docs/shared/acronyms.tex` follow too.

## Proposals to check

These are not owner decisions yet. They are settled at the owner checkpoint at the latest, because they end up inside the audio.

1. Shares: the 70% left after clean and competing talker splits evenly over the 9 categories, so babble falls from 20% to about 8%.
2. WHAM! `tt` is City noise in the test split, as in LibriVAD. ADR 0023 had reserved it for Test-unseen-noise, and it still never trains.
3. Stored codec shares, `p_narrow` and all chain probabilities are placeholders.
4. CallHome stays at 16 kHz with its 8 kHz content, unchanged.
5. Brouhaha is deferred, so the Common Voice baseline uses WADA-SNR alone.
6. DTX and CNG are deferred with packet loss, since they need per-frame flags from the bitstream that are unverified through ffmpeg.
7. 5% of speakers are reserved, with a 50-speaker threshold for borrowing babble from other languages.
8. Synthetic transients leave out typing and handling noise; this one is the owner's choice, but it removes items ADR 0023 listed.
9. Babble is exactly 6 talkers and SSN uses 12th-order LPC, following the LibriVAD paper (arXiv 2512.17281).
10. SCAFE exists only at 48 kHz and stays out, so Street has two scenes (SPSQUARE, STRAFFIC) where LibriVAD has three.
11. Because LibriVAD's `medium` beat its `large` on its own test but `large` won on VOiCES, the checkpoint compares a shorter rank prefix of Large against the full one before headline results.
12. WADA-SNR saturates at its table ends on about half or more of sampled Common Voice clips, and the resulting correction to speech power is at most about 0.11 dB. `baseline_snr_db` is therefore informational unless the owner asks for Brouhaha.
13. SWC masks an inner gap of 2 s or more between aligned words, and a word longer than 5 s, because labels would otherwise disagree with the audio. Of 30 sampled unmasked gaps per length, speech-level audio was found in 23 over 10 s, 5 at 5 to 10 s, 3 at 2 to 5 s and 2 at 1 to 2 s; `swc/german/U-Bahn_Berlin#2@202340` has a 14,240 ms gap at speech level. Words over 5 s are mostly not speech, such as an 8,920 ms word at -63 dB and a 55,830 ms word. Masking is safe because padding supplies ample non-speech.
14. A clip whose RMS over the word union is under −70 dBFS fails as a logged row, since its speech labels would sit on digital silence. `swc/german/Aortendissektion@914740` has one 3.39 s word at −83.6 dBFS, with 84.1% of its stored samples exactly zero. About 1 row in 5,000 was measured like this.
15. SWC's local alignment is sometimes wrong. On the first real slice, 9 of 138 sampled samples (6.5%, indicative: a lag search can overfit), all in train, fit their labels much better 0.7 to 3 s away in raw audio. The test split comes from the same aligner, so it may be affected too; that is not measured. `swc/german/Drei_Chinesen_mit_dem_Kontrabass@34270` has words at 5.21 to 5.55 s on digital silence labelled speech, while 2.6 to 3.6 s at about -21 dBFS is labelled non-speech. The -70 dBFS row check averages the whole word union, so it cannot catch this. Labels also lead the audio by roughly 10 to 30 ms in raw SWC (a word-edge fit gives about 12 ms, 60 s windows about 30 ms; by median the onset is 40 ms early and the offset 17 ms late), stable and below G, and not caused by this pipeline. Question for the owner: does SWC need a per-article or per-chunk quality bar like the MFA one (see [Quality bar](#quality-bar))?

## Left to the builder

- Module boundaries under `training/src/openvad_train/dataset/`.
- MFA batch size, job counts and temporary disk use.
- Noise suppression details, handset filter design and transient waveforms within the stated ranges.
- Exact shard and Parquet row group sizes near the stated targets.

## Deferred

- Packet loss with PLC, DTX and CNG, and the labels they need.
- EVS and music.
- Test-grid, Test-unseen-noise, benchmark preparation.
- The 500-clip hand check.
- CallHome forced alignment from TalkBank transcripts.
- Small's size and the `small` config.
- Mixtures beyond one per clip.
