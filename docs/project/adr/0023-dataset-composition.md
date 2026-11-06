# 0023: Dataset composition, splits and tiers

Status: Proposed

## Context

[0018](0018-datasets.md) lists the sources and their roles, [0021](0021-telephone-channel-simulation.md) the channel chain, [0022](0022-frame-labels.md) what counts as speech and [0025](0025-alignment-labelling.md) how forced-alignment labels are made.
This record fixes which sources and languages feed the generated training set, how they are treated, split and capped, and that the set comes as a Small and a Large tier.
The build is specified in the [dataset construction spec](../spec/dataset-construction.md).
A licence does not keep a source out of training ([0024](0024-training-data-licence-policy.md)).
Volume, language counts and per-speaker numbers are in [`docs/notes/datasets.md`](../../notes/datasets.md).

## Proposal

### Sources and treatment

The sources and their treatment (owner, 2026-11-03):

| Source | Treatment |
| --- | --- |
| Common Voice 27.0 | augmented (noise at a range of SNRs) and through the telephone channel of [0021](0021-telephone-channel-simulation.md) |
| MLS, as FLAC | augmented and telephone channel |
| FLEURS | augmented and telephone channel |
| SWC | augmented and telephone channel |
| LibriVAD `medium`, NonConcat and Concat | as generated upstream: train, validation and test splits (owner, 2026-11-06) |
| VoxPopuli, transcribed part | unmodified |
| AMI, individual headsets | unmodified |
| Speech Commands | unmodified |
| CallHome | unmodified: no added noise and no simulated channel; the mirror's 16 kHz files hold 8 kHz telephone audio and are used unchanged (`band = narrow_native`) |
| VOiCES | unmodified (owner, 2026-11-04) |

- VOiCES counts as ≈ 2.6 h of unique English speech; its re-recordings are far-field audio outside that count (owner, 2026-11-04).
- CallHome is not converted: it stays at 16 kHz with its 8 kHz content (proposed), and its `band` is `narrow_native`, so training adds no narrowband step ([0021](0021-telephone-channel-simulation.md)).
- AMI and CallHome: windows cut from the conversations at pauses, not padded; their speech ratio is their own.
- Common Voice noise is mixed relative to a baseline estimated with WADA-SNR ([`docs/notes/speech-quality.md`](../../notes/speech-quality.md)), since its clips carry device noise. Brouhaha is deferred, because it needs an isolated `numpy<2` environment (proposed).

### Storage and pointers

- Everything stored is 16 kHz mono FLAC (owner, 2026-11-06); nothing is stored at 8 kHz. Sources above 16 kHz are converted with ffmpeg, with the version recorded ([0009](0009-native-16khz-and-one-resampler.md)).
- Every clip of the pool is planned once (owner, 2026-11-06), with one mixture per clip.
- Constructed (augmented) audio lives in uncompressed tar shards under `data/`. Unmodified sources are never copied: their rows point into `datasets/raw` (owner, 2026-11-06). That covers VoxPopuli, AMI, CallHome, Speech Commands, VOiCES and LibriVAD.
- One Parquet row per sample holds a pointer relative to `$OPENVAD_DATA_DIR`, the labels on the 10 ms grid and the metadata. The files are split Parquet files with pointers, and one index maps each sample to its split and tier (owner, 2026-11-06). They are read and written asynchronously with Polars (owner, 2026-11-06).
- A build that stops early because the share is full is an interim set: usable for training and screening, but headline results wait for the final frozen set (owner, 2026-11-06).
- Train samples carry a stable random order, `rank = blake2(sample_id)`.

### Languages

- Languages are those whose speech events already exist as labels, or that the Montreal Forced Aligner (MFA) can label with a pretrained model (`*_mfa`, `*_cv` or VoxCommunis) (owner, 2026-11-03).
- Persian, Urdu, Meadow Mari, Arabic, Laz, Kabardian, Adyghe and Puno Quechua have no MFA model and are left out (owner, 2026-11-03).
- No language is held out: everything MFA can label is trained on, including Korean, Vietnamese, Kazakh and Hausa (owner, 2026-11-04).
- There is no unseen-language test and no per-language score table, so the thesis claims nothing about unseen languages; the owner expects a VAD to be language independent (owner, 2026-11-04).
- There is no minimum share of non-Indo-European speech; the owner is still constructing the dataset (owner, 2026-11-03).

### Cap and quotas

- At most 30 h per speaker in each dataset, so no speaker dominates a source (owner, 2026-11-03), to ensure speaker diversification and no overfitting (owner, 2026-11-06).
- There are no quotas per source or language for now; the owner wants to see the fetched pool first (owner, 2026-11-04).

### Labels

| Source | Labels |
| --- | --- |
| Common Voice, MLS, FLEURS, VoxPopuli | MFA with pretrained models ([0025](0025-alignment-labelling.md)); VoxPopuli `invalid` rows dropped; rows with `speaker_id` `None` kept, train only |
| SWC | its own word alignments; never-aligned stretches masked |
| LibriVAD | its own labels, MFA alignments of LibriSpeech (owner, 2026-11-04), used as they are: per-sample `int16` reduced to 10 ms by majority (proposed, without the gap rule) |
| AMI, individual headsets | manual word timings; crosstalk of other participants is non-speech; vocal sounds (laughter, coughs) are non-speech (owner, 2026-11-03) |
| CallHome | turns as given: the union of both sides is speech, so overlapping turns count as speech (proposed); ±0.25 s masked around every turn boundary of either side (owner, 2026-11-03) |
| Speech Commands | the single word aligned with MFA, exempt from the speaking-rate and first-20-ms drop rules (owner, 2026-11-06); the background-noise clips (0.11 h) are non-speech |
| VOiCES | the LibriSpeech source alignments, shifted by the measured delay (owner, 2026-11-04) |

- The gap rule of [0022](0022-frame-labels.md) applies to sources with word or phone timings; turn-based sources (CallHome, and Simsamu as a benchmark) use their turns as given (owner, 2026-11-04).
- The dominant-speaker task of [0022](0022-frame-labels.md) applies to every source, except as left open below for overlapping CallHome turns.
- Masked frames are a third frame class that the loss ignores ([0013](0013-bce-loss.md)) (owner, 2026-11-04).
- The drop rules of [0025](0025-alignment-labelling.md) apply per language and source.

### Splits

- Dev and test are a source's official split where it has one, otherwise about 10% each (owner, 2026-11-03).
  The table states what separation each source can guarantee.
- LibriSpeech `dev-clean` and `test-clean` are never read directly (owner, 2026-11-03); they reach validation and test through LibriVAD only, never train (owner, 2026-11-06).
- Dev and test are fixed and the same for both tiers.
- Reserved speakers: 5% of the speakers of each language, at least one, on train and on dev and test separately, feed babble and competing talkers only and are never foreground (proposed). FLEURS has no speaker IDs and reserves nothing. A language with under 50 speakers draws babble from any language.

| Source | Dev and test | Separation |
| --- | --- | --- |
| Common Voice | official dev and test; train is every validated clip whose speaker is in neither (owner, 2026-11-03) | speaker-disjoint, not sentence-disjoint |
| MLS | official | speaker-disjoint |
| AMI (full-corpus-ASR) | official | speaker-disjoint |
| Speech Commands | official; test is `testing_list.txt` (11,005 clips), not the separate 4,890-clip test set (owner, 2026-11-03) | speaker-disjoint |
| FLEURS | official | speaker-disjoint per the FLEURS paper; no speaker IDs, so not checkable |
| VoxPopuli | our own 80/10/10 by speaker ID, because the official splits share speakers; rows with `speaker_id` `None` (263.6 h) go to train only (owner, 2026-11-03) | speaker-disjoint for rows with an ID; `None` rows cannot be checked against dev and test, and the cap cannot apply to them |
| SWC | 80/10/10 by reader, by aligned hours | by reader |
| CallHome | 80/10/10 by call | by call; no speaker IDs across calls, so not checkable by speaker |
| VOiCES | see below | speaker-disjoint |
| LibriVAD | train: its `medium` train split, Large only. Validation: `medium` val generated from `dev-clean` (about 27.4 h NonConcat, 35.2 h Concat). Test: `medium` test from `test-clean`, as generated, and still the benchmark of [0020](0020-evaluation-benchmarks.md). Validation and test are never trained on (owner, 2026-11-06) | by LibriSpeech split |

VOiCES speakers come from LibriSpeech, which sets their split (owner, 2026-11-03):

- the 12 from `dev-clean` are dropped;
- the 64 from `train-clean-100` go to train only, since LibriVAD has their speech;
- the 224 from `train-clean-360` are split 80/10/10 by speaker; none comes from `test-clean`.

### Tiers

- There will be a Small and a Large set (owner, 2026-11-06).
- Each has a train, a validation (dev) and a test split (owner, 2026-11-06).
- Small is a subset of Large (owner, 2026-11-06).
- The size of Small is not decided (owner, 2026-11-06). Small is a rank cut-off over everything except LibriVAD, so it is a strict subset of Large and leaves out LibriVAD `medium`; it sets `in_small` only, and no audio is rebuilt.
- [0017](0017-experiment-budget.md) says what each tier is used for, including that LibriVAD `medium` train reaches training only through Large and that LibriVAD `small` train runs apart from OpenVAD Small.

### Structure and acoustic shares (augmented sources)

Single utterances are padded to about 33% speech, keeping this record's earlier choice (Kretzschmar 2023: best at 33%, overfitting at 50%) (owner, 2026-11-06).
Common Voice's natural clip edges count towards the padding. Speech goes at a random offset in one block.
Clean padding is Gaussian dither at −90 to −70 dBFS (proposed), so the model cannot detect digital silence.
MLS and SWC are split at word gaps longer than G into chunks of at most 6.6 s speech, each padded to 10 to 20 s (proposed).

The conditions are LibriVAD's nine noise categories plus clean and competing talker (owner, 2026-11-06). Music is not used (owner, 2026-11-06).

| Condition | Share (proposed) | Source |
| --- | --- | --- |
| Clean | 15% | |
| Competing talker | 15% | [0022](0022-frame-labels.md) |
| Babble | 70/9 % | 6 energy-normalised talkers summed, as LibriVAD; same language from reserved speakers; Japanese and Mandarin: same language with p = 0.5, any language otherwise (owner, 2026-11-03) |
| SSN | 70/9 % | Gaussian white noise through a 12th-order linear-predictive all-pole filter fitted to one reserved speaker, as LibriVAD |
| City | 70/9 % | WHAM!: `tr` train, `cv` dev, `tt` test (the `tt` use is proposed); one random channel per draw |
| Domestic, Nature, Office, Public, Street, Transport | 70/9 % each | DEMAND scenes by first letter (D, N, O, P, S, T), as LibriVAD, one random channel per draw: channels 1 to 12 train, 13 and 14 validation, 15 and 16 test (owner, 2026-11-06) |

- The 70% left after clean and competing talker splits evenly over the nine categories, so babble falls from 20% to about 8% (proposed).
- Noise comes from our pools, not LibriVAD's noise files (owner, 2026-11-06). DEMAND is split by channel, replacing the one shared pool; WHAM! `tt` never trains.
- MUSAN noise files are sorted into the nine categories by keyword, so its 6.2 h add variety to the thin DEMAND categories (owner, 2026-11-06). A committed table holds the result of a one-time keyword pass plus manual review, and files that fit no category are listed as unused. MUSAN contributes noise only, and is shared across splits.
- SNR is uniform over {−5, 0, 5, 10, 15, 20} dB, LibriVAD's grid (owner, 2026-11-06), replacing the bimodal ranges. It is computed from speech RMS over speech-active frames and noise RMS over the same span, then applied to the noise over the whole sample.
- Transients are synthetic clicks and DTMF only (owner, 2026-11-06), overlaid on any sample with p = 0.3 and mixed by peak ratio at 0 to −18 dB. Typing and handling noise are dropped.
- The channel chain follows [0021](0021-telephone-channel-simulation.md).
- The manifest holds one Parquet row per sample with: source IDs and offsets, source licences, condition, SNR or TIR, channel chain, bitrates, delays and seed. The loss pattern is absent, because packet loss is deferred ([0021](0021-telephone-channel-simulation.md)).

### Cutting into samples (proposed)

| Source | Cut |
| --- | --- |
| Common Voice, FLEURS | one clip, one sample, padded |
| MLS, SWC | chunks, see above |
| VoxPopuli | as annotated; above 30 s split at gaps; not padded |
| AMI, CallHome | cut at non-speech gaps to at most 30 s; not padded |
| Speech Commands | one clip, one sample; not padded |
| VOiCES | whole wav, every re-recording; distractor-only noise files not used |
| LibriVAD `medium` | every generated wav, one row each; each utterance appears 54 times by design |

### Test pools

[0020](0020-evaluation-benchmarks.md) owns Test-broad, Test-grid and Test-unseen-noise, and the benchmarks.

## Consequences

- Results are reported per dataset (owner, 2026-11-03): no macro averages over languages (owner, 2026-11-03), no per-language results and no per-family headline (owner, 2026-11-04).
- Common Voice dominates the pool, at about 79% of the train hours after the 30 h cap (≈ 16,830 of ≈ 21,390 h, partly estimated until its fetch ends); nothing caps it per source or language for now, so the tiers may reflect that.
- Which machines may render and train on these sources is limited by [0024](0024-training-data-licence-policy.md).

## Open questions

- The tier sizes: Large is everything, and the size of Small is open.
- What the dominant-speaker task means for overlapping CallHome turns, which are used as given.
- Whether LibriVAD's own labels follow the gap rule of [0022](0022-frame-labels.md) (not checked).
- Whether the shares, the WHAM! `tt` use in the test split, the cutting rules and the reserved-speaker share stay as proposed; they are settled at the owner checkpoint of the [dataset construction spec](../spec/dataset-construction.md).
- CallHome forced alignment with TalkBank's transcripts, once TalkBank access exists ([0018](0018-datasets.md)).

## References

- Kretzschmar 2023
