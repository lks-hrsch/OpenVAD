# 0018: Datasets

Status: Open

## Context

Data must be public and reproducible ([0004](0004-project-scope.md)).
This record says which public sources OpenVAD uses, in which role and why; [0023](0023-dataset-composition.md) says how the training pool is built from them.
Licences are recorded for every source but do not decide what may be trained on ([0024](0024-training-data-licence-policy.md)).
Per-source facts are in [`docs/notes/datasets.md`](../../notes/datasets.md), fetch details and licences in [`data/README.md`](../../../data/README.md).
In the Scale column, sizes on disk come from the manifests on the data share, hours from the notes (≈ marks an estimate), and the Common Voice extraction state is that of 2026-11-06, while its fetch is still running.

## Findings

### Sources used

| Source | Role | Reason | Scale |
| --- | --- | --- | --- |
| Common Voice Scripted Speech 27.0 | train, dev, test | the only public source with many speakers and transcripts outside Europe; its clips carry device noise, so it never enters a fixed-SNR test | partial: 27 of 53 locales extracted; ≈ 16,580 h train after the 30 h cap (15,575.3 h measured in the 27, ≈ 1,000 h estimated for the rest); 670 GB download, of which 558 GB is extracted |
| MLS, 7 languages | train, dev, test | read audiobooks with speaker-disjoint splits; FLAC, because the [0021](0021-telephone-channel-simulation.md) chain must be the only codec stage and Opus would put a hidden codec before every sample | 1,445,342 segments and 562 speakers in train, 13,974 segments in dev, 13,933 in test; 2,178 h train after the 30 h cap; 377.2 GB on disk |
| FLEURS, 42 languages | train, dev, test | the same sentences in every language, so content is controlled across languages | 397.6 h train, 49.3 h dev, 115.8 h test; 129.4 GB on disk |
| SWC, English, German, Dutch | train, dev, test | long read articles with word alignments; only aligned words are speech, and unaligned stretches are unknown | ≈ 347 h aligned train after the 30 h cap; 38.8 GB on disk |
| VoxPopuli, transcribed part, 13 languages | train, dev, test | semi-spontaneous parliament speech | ≈ 1,464 h train; 77.5 GB on disk |
| AMI, individual headsets | train, dev, test | spontaneous meetings with manual word timings; the wearer is dominant and the others are crosstalk, which matches [0022](0022-frame-labels.md) | 80.6 h train, 9.7 h dev, 9.1 h test; 46.1 GB on disk |
| CallHome, TalkBank mirror, five languages | train, dev, test | real, unscripted telephone conversations between native speakers, with 8 kHz originals | 660 calls, 99.6 h; 11.2 GB on disk |
| Speech Commands v0.02 | train, dev, test | one-second keywords from many speakers; labels per clip only | 105,829 clips, 2,618 speakers; 3.5 GB on disk |
| VOiCES devkit | train, dev, test | far-field re-recordings of LibriSpeech, so not independent of it; the full release only re-records more of the same utterances | ≈ 84 h of re-recordings of ≈ 2.6 h unique speech, ≈ 235 h of room noise; 37.7 GB on disk |
| LibriVAD `medium` | train split as a pool source, validation split in validation, test split in test and as a benchmark; validation and test are never trained on (owner, 2026-11-06) | a public VAD training set; the unmodified build at the pinned revision is kept (owner, 2026-11-01) | 546 h NonConcat, 680 h Concat in train; every 10th utterance of `train-clean-100` with 54 mixtures each, so about 10 h of unique speech; validation (`dev-clean`) about 27.4 h NonConcat and 35.2 h Concat, not generated yet; 210.4 GB on disk for train and test |
| LibriVAD `small` | train split for screening, test split as the paper-comparison row of the LibriVAD benchmark (owner, 2026-11-04) | the size the LibriVAD paper reports on | 15.0 GB on disk |
| LibriSpeech `train-clean-100`, `dev-clean`, `test-clean` | source of LibriVAD and Picovoice only; `dev-clean` and `test-clean` reach validation and test through LibriVAD, never directly, and `test-clean` is benchmark audio | LibriVAD builds its test split from `test-clean` and its validation split from `dev-clean`, so using them directly would leak the evaluation | 7.3 GB on disk |
| LibriSpeech alignments, Zenodo 2619474 | labels for VOiCES and the `test-clean` benchmark only (owner, 2026-11-04) | public CC BY 4.0 alignments, separate from LibriVAD's CC BY-NC-SA repository | 1.7 GB on disk |
| DEMAND | noise, split by channel: 1 to 12 train, 13 and 14 validation, 15 and 16 test (owner, 2026-11-06) | one random channel per draw; still seen noise in the Picovoice benchmark | 2.6 GB on disk |
| MUSAN | noise only, one pool for all splits (owner, 2026-11-04); music is not used (owner, 2026-11-06) | the 6.2 h of noise add variety to the thin DEMAND categories after a keyword sort into the nine categories of [0023](0023-dataset-composition.md) (owner, 2026-11-06); its speech has no word labels; babble comes from the train speakers of the pool (owner, 2026-11-04) | 12.6 GB on disk, unused speech and music included |
| WHAM! noise | `tr` train, `cv` dev, `tt` test (City noise) | its splits share no recording or location; LibriVAD's City noise is cut from the same splits, so `tt` never trains | `tr` 58.0 h, `cv` 14.7 h, `tt` 9.0 h; 37.7 GB on disk |
| BUT ReverbDB | room impulse responses for the room step of [0021](0021-telephone-channel-simulation.md) (owner, 2026-11-06) | real rooms instead of simulated ones; the 37 rooms of both sets are ranked by a hash of the room, the first tenth validation, the next test and the rest train, so no room is in two splits | CC BY 4.0; release 19_06, 9.3 GB archive of which only the RIR wavs of 9 rooms are kept; fetcher written, not fetched to the share yet |
| OpenSLR 28 (RIRS_NOISES) | room impulse responses, as BUT ReverbDB (owner, 2026-11-06) | the same; only its 325 real RIRs in 28 rooms (AIR, REVERB 2014, RWCP), not the simulated ones | Apache-2.0; 1.3 GB archive; fetcher written, not fetched to the share yet |
| AVA-Speech | benchmark | independent in-the-wild audio with human labels | 151 videos; 19.5 GB on disk |
| Fearless Steps Phase 4 | benchmark | real narrowband channels with human labels | not fetched (registration) |
| Picovoice voice-activity-benchmark | benchmark | published ROC curves | 2.3 GB on disk |
| Simsamu | benchmark, 8 kHz condition (owner, 2026-11-01) | target domain, French telephone; only the caller side is a telephone channel | 61 calls, 3.22 h annotated; 56.8 MB on disk |

## Open questions

- How the CallHome mirror's audio relates to the original two-channel 8 kHz calls, and whether TalkBank's CHAT transcripts line up with it closely enough for forced alignment.
- Why Buckeye, ESC-50, LibriMix noise and AudioSet stay out. Their caveats (non-commercial; AudioSet decays as YouTube videos disappear) do not exclude them by licence ([0024](0024-training-data-licence-policy.md)).
- Why English MLS stays out. LibriSpeech covers read English and MLS English train shares speakers with LibriSpeech train, while LibriSpeech now reaches training only through LibriVAD; the fetcher only omits it.

## References

- LibriVAD 2025 (SSR)
- MLS (Pratap et al. 2020, arXiv 2012.03411)
- WHAM! (Wichern et al. 2019, arXiv 1907.01160)
