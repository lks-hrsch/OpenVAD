# 0023: Dataset composition, splits and tiers

Status: Proposed

## Context

[0018](0018-datasets.md) lists the sources and their roles, [0021](0021-telephone-channel-simulation.md) the channel chain, [0022](0022-frame-labels.md) what counts as speech and [0025](0025-alignment-labelling.md) how forced-alignment labels are made.
This record fixes which sources and languages feed the generated training set, how they are treated, split and capped, and that the set comes as a Small and a Large tier.
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
| LibriVAD `medium`, NonConcat and Concat | as generated upstream |
| VoxPopuli, transcribed part | unmodified |
| AMI, individual headsets | unmodified |
| Speech Commands | unmodified |
| CallHome | unmodified: no added noise and no simulated channel, stored at 8 kHz |
| VOiCES | unmodified (owner, 2026-11-04) |

- VOiCES counts as ≈ 2.6 h of unique English speech; its re-recordings are far-field audio outside that count (owner, 2026-11-04).
- CallHome is converted to 8 kHz with the OpenVAD decimator, as every 8 and 16 kHz conversion is ([0009](0009-native-16khz-and-one-resampler.md)) (owner, 2026-11-04).
- AMI and CallHome: windows cut from the conversations at pauses, not padded; their speech ratio is their own.
- Common Voice noise is mixed relative to an estimated baseline (Brouhaha and WADA-SNR, [`docs/notes/speech-quality.md`](../../notes/speech-quality.md)), since its clips carry device noise.

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
| LibriVAD | its own labels, MFA alignments of LibriSpeech (owner, 2026-11-04) |
| AMI, individual headsets | manual word timings; crosstalk of other participants is non-speech; vocal sounds (laughter, coughs) are non-speech (owner, 2026-11-03) |
| CallHome | turns as given, ±0.25 s masked around each boundary (owner, 2026-11-03) |
| Speech Commands | the single word aligned with MFA; the background-noise clips (0.11 h) are non-speech |
| VOiCES | the LibriSpeech source alignments, shifted by the measured delay (owner, 2026-11-04) |

- The gap rule of [0022](0022-frame-labels.md) applies to sources with word or phone timings; turn-based sources (CallHome, and Simsamu as a benchmark) use their turns as given (owner, 2026-11-04).
- The dominant-speaker task of [0022](0022-frame-labels.md) applies to every source, except as left open below for overlapping CallHome turns.
- Masked frames are a third frame class that the loss ignores ([0013](0013-bce-loss.md)) (owner, 2026-11-04).
- The drop rules of [0025](0025-alignment-labelling.md) apply per language and source.

### Splits

- Dev and test are a source's official split where it has one, otherwise about 10% each (owner, 2026-11-03).
  The table states what separation each source can guarantee.
- LibriSpeech `dev-clean` and `test-clean` never enter dataset creation (owner, 2026-11-03).
- Dev and test are fixed and the same for both tiers.

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
| LibriVAD | its train split only; its test split is a benchmark | |

VOiCES speakers come from LibriSpeech, which sets their split (owner, 2026-11-03):

- the 12 from `dev-clean` are dropped;
- the 64 from `train-clean-100` go to train only, since LibriVAD has their speech;
- the 224 from `train-clean-360` are split 80/10/10 by speaker; none comes from `test-clean`.

### Tiers

- There will be a Small and a Large set (owner, 2026-11-06).
- Each has a train, a validation (dev) and a test split (owner, 2026-11-06).
- Small is a subset of Large (owner, 2026-11-06).
- [0017](0017-experiment-budget.md) says what each tier is used for, including that LibriVAD `medium` train reaches training only through Large and that LibriVAD `small` train runs apart from OpenVAD Small.

### Structure and acoustic shares (augmented sources)

Single utterances are padded with noise or silence to ≈ 33% speech (Kretzschmar 2023: best at 33%, overfitting at 50%).
Common Voice's natural clip edges count towards the padding.

| Condition | Share | Source |
| --- | --- | --- |
| Clean | 15% | |
| Ambient noise | 40% (owner, 2026-11-03) | DEMAND (one random channel per draw), MUSAN noise, WHAM! `tr` (one random channel per draw) |
| Music | 10% | MUSAN music |
| Babble | 20% | same language from train speakers never used as foreground (owner, 2026-11-04); Japanese and Mandarin: same language with p = 0.5, any language otherwise (owner, 2026-11-03) |
| Competing talker | 15% | [0022](0022-frame-labels.md) |

- WHAM! `tr` is train noise, `cv` dev noise and `tt` the unseen-noise test noise; DEMAND and MUSAN are shared across splits (owner, 2026-11-03).
- MUSAN contributes noise and music only (owner, 2026-11-04).
- Transient events (clicks, typing, handling noise, DTMF) are overlaid on any sample with p = 0.3, mixed by peak ratio at 0 to −18 dB.
- SNR is bimodal: easy 5 to 20 dB (60%), hard −5 to 5 dB (40%).
- The channel chain follows [0021](0021-telephone-channel-simulation.md).
- Everything is precomputed as FLAC with labels on the 10 ms grid, plus one manifest entry per sample: source IDs and offsets, source licences, condition, SNR or TIR, channel chain, bitrates, loss pattern, delays and seed.

### Test pools

[0020](0020-evaluation-benchmarks.md) owns Test-broad, Test-grid and Test-unseen-noise, and the benchmarks.

## Consequences

- Results are reported per dataset (owner, 2026-11-03): no macro averages over languages (owner, 2026-11-03), no per-language results and no per-family headline (owner, 2026-11-04).
- Common Voice dominates the pool, at about 78% of the train hours after the 30 h cap (≈ 16,830 of ≈ 21,480 h, partly estimated until its fetch ends); nothing caps it per source or language for now, so the tiers may reflect that.
- Which machines may render and train on these sources is limited by [0024](0024-training-data-licence-policy.md).

## Open questions

- The tier sizes and how Small is drawn, including whether it leaves out LibriVAD `medium`.
- How VoxPopuli, Speech Commands and VOiCES are cut into samples, and whether any of them is padded.
- What the dominant-speaker task means for overlapping CallHome turns, which are used as given.
- Whether LibriVAD's own labels follow the gap rule of [0022](0022-frame-labels.md) (not checked).
- CallHome forced alignment with TalkBank's transcripts, once TalkBank access exists ([0018](0018-datasets.md)).

## References

- Kretzschmar 2023
