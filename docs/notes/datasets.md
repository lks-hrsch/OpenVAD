# Datasets

Measured numbers and source facts for every dataset OpenVAD uses.
State of 2026-11-07.

Decisions are in the ADRs: sources and roles in [0018](../project/adr/0018-datasets.md), experiment stages and the dev subset for early stopping in [0017](../project/adr/0017-experiment-budget.md), benchmarks, test pools and thresholds in [0020](../project/adr/0020-evaluation-benchmarks.md), composition, splits and labels in [0023](../project/adr/0023-dataset-composition.md), licence policy and data location in [0024](../project/adr/0024-training-data-licence-policy.md), forced-alignment labelling in [0025](../project/adr/0025-alignment-labelling.md).
What each fetcher downloads, and each licence, is in [`data/README.md`](../../data/README.md).
Aligners and the model per language are in [`forced-alignment.md`](forced-alignment.md), per-clip quality tools in [`speech-quality.md`](speech-quality.md).

## Rules

- At most 30 h per speaker in each dataset, to ensure speaker diversification and no overfitting ([0023](../project/adr/0023-dataset-composition.md)).
- There will be a Small and a Large set, each with a train, validation (dev) and test split, and Small is a subset of Large ([0023](../project/adr/0023-dataset-composition.md)).
- Every other rule is in an ADR: public data, no redistribution, where the data lives and what never enters training in [0024](../project/adr/0024-training-data-licence-policy.md); splits per source, the CallHome mask and the VOiCES speaker drop in [0023](../project/adr/0023-dataset-composition.md); 10 ms frames and the dominant speaker in [0022](../project/adr/0022-frame-labels.md); results per dataset only in [0023](../project/adr/0023-dataset-composition.md) and [0020](../project/adr/0020-evaluation-benchmarks.md).

## Use per dataset

What each source feeds; the benchmarks and test pools are defined in [0020](../project/adr/0020-evaluation-benchmarks.md), the reference systems in [0003](../project/adr/0003-isolated-benchmark-adapters.md).

| Dataset | Train | Dev | Test | Benchmark or noise |
| --- | --- | --- | --- | --- |
| LibriSpeech | source of LibriVAD and Picovoice only | `dev-clean` reaches validation through LibriVAD only | `test-clean` reaches test through LibriVAD only | benchmark: `test-clean` (test-grid); never read directly |
| LibriSpeech alignments (Zenodo 2619474) | labels for VOiCES | labels for VOiCES | labels for VOiCES | labels for the `test-clean` benchmark; `train-clean-360` alignments kept for the VOiCES speakers, though that audio is not fetched |
| Common Voice 27.0 | validated clips of speakers outside dev and test, augmented | official dev | official test | |
| SWC 2.0 | by reader 80%, augmented | 10% | 10% | |
| MLS | official train, augmented | official dev | official test | |
| VoxPopuli (transcribed) | own split by speaker, 80%, unmodified; rows without a speaker ID too | 10% | 10% | |
| AMI (IHM) | full-corpus-ASR train, unmodified | official | official | |
| CallHome (TalkBank mirror) | by call, 80%, unmodified, 16 kHz files with 8 kHz content (`narrow_native`) | 10% | 10% | |
| FLEURS | official train, augmented | official dev | official test | |
| Speech Commands v0.02 | official, unmodified | official | `testing_list.txt` (11,005 clips) | |
| VOiCES devkit | by speaker, 80%, unmodified; 64 `train-clean-100` speakers train only, 12 `dev-clean` speakers dropped | 10% | 10% | |
| MUSAN | noise only, sorted into the nine categories by keyword | noise only | noise only | noise, every file; music is not used |
| DEMAND | noise, channels 1 to 12 | noise, channels 13 and 14 | noise, channels 15 and 16 | noise |
| WHAM! noise | `tr` (City) | `cv` (City) | `tt` (City; proposed, as in LibriVAD) | noise |
| BUT ReverbDB, OpenSLR 28 | room impulse responses, rooms split 80/10/10 by hash | same | same | |
| LibriVAD `medium` | train split as generated (NonConcat and Concat), only through Large | val split generated from `dev-clean`, validation only | test split as generated, never trained on | benchmark: test split |
| LibriVAD `small` | train split, screening only | | | benchmark: test split (paper-comparison row of the LibriVAD benchmark) |
| AVA-Speech | | | | benchmark |
| Fearless Steps Phase 4 | | | | benchmark; dev split for a second operating point |
| Picovoice voice-activity-benchmark | | | | benchmark |
| Simsamu | | | | benchmark (8 kHz) |

Dev conditions are drawn like train, with WHAM! `cv` as the City noise and DEMAND channels 13 and 14.
Babble in dev and test comes from dev and test speakers.
Noise categories and shares are in [0023](../project/adr/0023-dataset-composition.md); the room impulse responses are in [Per-source facts](#room-impulse-responses).

## Pool

Train hours are audio hours, speech only where marked (AMI, CallHome, LibriVAD, Speech Commands, SWC); Common Voice, MLS and SWC are after the 30 h cap, FLEURS has no speaker IDs to cap, and the rest are under it.
Speakers count all splits, train only where marked.
Numbers marked ≈ are estimates.
The family and source-type tables use these train hours.

| Source | Languages | Speakers (all splits; train where marked) | Train hours (audio; speech where marked) | Rendered |
| --- | --- | --- | --- | --- |
| Common Voice 27.0 | 53 locales | ≈ 330,000 | ≈ 16,580 h after the cap: 15,575.3 h measured in 27 locales, ≈ 1,000 h estimated for the other 26 | augmented |
| MLS | DE 819, FR 497, ES 318, IT 238, NL 135, PT 104, PL 67 (after the cap) | 562 (train) | 2,178 h | augmented |
| VoxPopuli transcribed | 13: EN 477, DE 231, FR 184, ES 134, PL 90, IT 74, RO 72, HU 51, CS 50, NL 43, HR 34, FI 22, ET 2 | ≈ 2,380 IDs, plus rows without an ID | ≈ 1,464 h: 80% of the rows with a speaker ID, plus the 263.6 h of `None` rows (EN 201.0, FR 28.6, DE 16.3; train only) | as is |
| SWC | EN ≈ 147, DE ≈ 166, NL ≈ 34 (after the cap) | ≈ 900 reader names, plus 25.6 h without one | ≈ 347 h aligned speech: 80% of the 434.4 h left under the cap | augmented |
| FLEURS | 42 languages | unknown (no speaker IDs) | 397.6 h (measured) | augmented |
| LibriVAD `medium` | EN | 251 (train) | ≈ 10 h unique speech (every 10th utterance of `train-clean-100`, 54 mixtures each) | 546 h NonConcat, 680 h Concat |
| AMI IHM | EN | 152 (train) | ≈ 76 h speech (word-timing union per headset) | as is |
| CallHome | EN, DE, ES, JA, ZH | ≈ 528 calls (80% of 660), ≈ 1,245 speaker labels (train) | 71 h speech in 80 h audio, before the ±0.25 s mask | as is |
| Speech Commands v0.02 | EN | 2,618 (2,112 train) | ≈ 15 h speech (estimate) in ≤ 23.6 h audio | as is |
| VOiCES devkit | EN | 300 (re-recorded LibriSpeech), 288 after the drop | ≈ 2.6 h unique (600 utterances; ≈ 84 h of re-recordings in the devkit) | as is |
| **Total** | **≈ 53 languages** | | **≈ 21,140 h** | |

- The total is the sum of the rows.
- About 53 languages is a union: Common Voice has 51 distinct languages in its 53 locales, and FLEURS adds Bosnian and Croatian.
  The 53 locales and the ≈ 53 languages are different counts that happen to be equal.
- The estimate for the 26 locales not yet extracted applies the ratio of the 19 measured locales under 500 h to their 1,273.5 h in the release statistics: 81.9% in the train pool, 78.8% under the cap. It is replaced by measurements as the locales arrive, by about 11 November.

### Effect of the 30 h cap

| Source | Before | After | Removed | Who passes 30 h |
| --- | --- | --- | --- | --- |
| Common Voice, 27 locales measured | 17,329.2 h | 15,575.3 h | 1,753.9 h | 49 speakers in 15 of the 27 locales; per locale in [Common Voice](#common-voice) |
| Common Voice, all 53 locales | ≈ 18,370 h: 17,329.2 h measured, ≈ 1,040 h estimated | ≈ 16,580 h | | |
| MLS | 6,027.0 h | 2,178 h | 3,849.5 h | a few narrators; Dutch keeps 135 of 1,554 h |
| SWC, aligned speech | 516.9 h | 434.4 h | 82.4 h | two readers, see [SWC](#swc) |

LibriVAD, AMI, CallHome, VoxPopuli and Speech Commands are under the cap (VoxPopuli's `None` rows have no speaker and are not capped); FLEURS has no speaker IDs to cap.

### Language families

Information only: there is no per-family headline, and there is no minimum share of non-Indo-European speech ([0023](../project/adr/0023-dataset-composition.md)).

| Family | Hours | Share | Main sources |
| --- | --- | --- | --- |
| Indo-European | ≈ 15,040 | 71% | Common Voice (English, Catalan, German, Belarusian, French, Spanish, ...), MLS, VoxPopuli, SWC, FLEURS, LibriVAD, AMI, CallHome English, German and Spanish, Speech Commands, VOiCES |
| Atlantic–Congo | ≈ 2,510 | 12% | Common Voice Kinyarwanda, Luganda, Swahili; FLEURS Luganda, Swahili |
| Turkic | ≈ 850 | 4% | Common Voice Uyghur, Bashkir, Turkish, Uzbek, Kyrgyz, Tatar, Chuvash, Kazakh; FLEURS Kyrgyz, Turkish, Uzbek, Kazakh |
| Sino-Tibetan | ≈ 575 | 3% | Common Voice Mandarin and Cantonese, CallHome Mandarin, FLEURS |
| Afroasiatic | ≈ 460 | 2% | Common Voice Kabyle and Hausa; FLEURS Hausa |
| Basque | ≈ 430 | 2% | Common Voice |
| Japonic | ≈ 370 | 2% | Common Voice, CallHome, FLEURS |
| Kartvelian | ≈ 160 | 1% | Common Voice Georgian, FLEURS |
| Uralic | ≈ 265 | 1% | Common Voice and VoxPopuli Hungarian, Estonian, Finnish; FLEURS |
| Dravidian, Kra–Dai, NW Caucasian, Mongolic | ≈ 455 | 2% | Common Voice Tamil, Thai, Abkhaz, Mongolian; FLEURS Tamil, Thai, Mongolian |
| Austroasiatic | ≈ 16 | < 0.1% | Common Voice and FLEURS Vietnamese |
| Koreanic | ≈ 10 | < 0.1% | Common Voice and FLEURS Korean |
| Tupian | ≈ 4 | < 0.1% | Common Voice Guarani |

- Rows are rounded and built per language after the cap; they sum to ≈ 21,145 h, ≈ 5 h above the pool total, and the rounded shares to 100%.
- Each row adds the measured Common Voice hours, the estimate of 78.8% of the release hours for locales not yet extracted, the measured FLEURS train hours, the VoxPopuli hours per language and CallHome speech split by the hours of its calls.
- Non-Indo-European ≈ 6,105 h, 29% of the pool.
- Families by current speakers ([Wikipedia, List of language families](https://en.wikipedia.org/wiki/List_of_language_families)): Indo-European (586 languages, 3.24 billion speakers), Sino-Tibetan (1.39 billion), Atlantic–Congo (500 million), Afroasiatic (499 million), Austronesian (326 million), Dravidian (253 million), Turkic (180 million), Japonic (129 million), Austroasiatic (116 million), Kra–Dai (82 million), Koreanic (77 million), Nilotic, Mande, Uralic (21 million), Mongolic, Quechuan; Basque is an isolate.

### Source types

| Type | Sources | Share |
| --- | --- | --- |
| Crowd-sourced read sentences | Common Voice, Speech Commands | ≈ 78% |
| Read audiobooks and Wikipedia | MLS, SWC, FLEURS, LibriVAD | ≈ 14% |
| Parliament, semi-spontaneous | VoxPopuli | ≈ 7% |
| Meetings and telephone conversations | AMI, CallHome | ≈ 0.7% |
| Re-recorded far-field read speech | VOiCES | < 0.1% |

- Common Voice is 78% of the train hours.
- Outside Europe, Common Voice is the only public source with many speakers and transcripts, so the non-Indo-European hours are almost all Common Voice, and language family and source type are confounded.
- Conversational and telephone speech is a small share of the pool (AMI and CallHome, about 0.7%), although the deployment target is telephone audio; the channel simulation of [0021](../project/adr/0021-telephone-channel-simulation.md) runs on augmented samples only, its mix includes a no-channel group, and it does not make them conversational.

## Common Voice

- Scripted Speech is distributed only through the Mozilla Data Collective (MDC) since October 2025. Release 27.0 has 295 locales, all CC0-1.0 and free, 0.94 TB together; 73 have at least 20 validated hours, 39 at least 100 h (from `cv-corpus-27.0-2026-09-11.json` in `common-voice/cv-dataset` at the pinned commit).
- The fetcher takes the 53 locales that have an MFA model (670 GB) and records the validated hours and speakers per locale in its manifest.
- Licence: CC0-1.0. The dataset terms add two forbidden uses: re-hosting or re-sharing the dataset, and any attempt to identify speakers. The MDC terms (updated 2026-10-28) grant rights only through that licence and forbid redistribution; they do not restrict machine-learning training or commercial use of the data, and "model outputs" must contain no personal data. Audio, also modified (mixed, noised, resampled), is never published, only code and our own labels.
- Download: after accepting each dataset's terms on the MDC site, the official API returns a short-lived URL, so the fetch is scripted.
- Validated clips: English 1,929,854 (2,723.8 h), German 953,365 (1,385.7 h), French 792,011 (1,095.3 h). Official splits (clips): English train 1,147,829, dev and test 16,403 each; German 624,800 and 16,209; French 621,982 and 16,207.
- The release statistics give 19,954 validated hours for the 53 locales, but as validated clips times the mean duration of all clips, so they are an estimate and not an upper bound: English 2,826.8 h against 2,723.8 h measured, Catalan 3,392.4 h against 3,465.7 h. Across the 27 measured locales the two agree within 0.1% (18,680.8 h and 18,683.8 h).
- The splits come from CorporaCreator: every speaker lands in exactly one split, each sentence is kept once, and the speakers with the fewest clips fill test, then dev, so every heavy speaker is in train.
- The trap in the official splits: because each sentence is kept once in train, for many languages the official train split is a small part of the validated clips: Japanese 24 h of 373 h, Persian 33 of 373, Russian 39 of 257, Pashto 252 of 3,530.
  A VAD does not need sentence-disjoint splits, only speaker-disjoint ones, so [0023](../project/adr/0023-dataset-composition.md) trains on every validated clip whose speaker is in neither the official dev nor test split.
  The fetcher's manifest records the resulting train pool and the hours left under the 30 h per-speaker cap.
- Speaker imbalance cannot be read from the release statistics; the manifest records the ten largest train speakers.
- Common Voice is its own device source, not clean speech: the clips carry device noise, reverberation and clicks, at an unknown baseline.
- Spontaneous Speech 5.0 (CC0-1.0, the same MDC terms) has one transcript per clip and no timestamps: English 18.6 h (8.4 h validated; train 5.0 h, dev 1.8 h, test 1.6 h), German 1.5 h and French 2.4 h without splits.
- Hours per locale, measured 2026-11-06 and 2026-11-07 from the extracted TSVs by the rule of `split_stats` in `fetch/commonvoice.py`; the other 26 locales are not extracted yet. The train pool is every validated clip of a speaker in neither dev nor test.

  | Locale | Validated h | Train pool h | Pool speakers | Under the cap h | Speakers over 30 h | Largest speaker h |
  | --- | --- | --- | --- | --- | --- | --- |
  | Catalan (`ca`) | 3,465.7 | 3,409.5 | 26,467 | 2,636.1 | 13 | 272.5 |
  | English (`en`) | 2,723.8 | 2,662.6 | 65,935 | 2,592.7 | 1 | 99.9 |
  | Kinyarwanda (`rw`) | 2,034.0 | 1,975.1 | 605 | 1,794.8 | 12 | 64.5 |
  | Belarusian (`be`) | 1,808.7 | 1,755.6 | 3,729 | 1,441.5 | 5 | 293.9 |
  | German (`de`) | 1,385.7 | 1,327.1 | 11,278 | 1,199.2 | 3 | 116.9 |
  | French (`fr`) | 1,095.3 | 1,042.1 | 11,764 | 1,002.6 | 2 | 56.8 |
  | Spanish (`es`) | 619.9 | 565.7 | 7,914 | 548.3 | 1 | 47.5 |
  | Kabyle (`kab`) | 568.0 | 523.2 | 337 | 443.1 | 4 | 62.7 |
  | Basque (`eu`) | 482.6 | 429.2 | 3,438 | 429.2 | 0 | 17.0 |
  | Luganda (`lg`) | 435.6 | 389.2 | 140 | 369.0 | 2 | 41.2 |
  | Uyghur (`ug`) | 460.5 | 369.9 | 182 | 356.8 | 1 | 43.1 |
  | Japanese (`ja`) | 386.7 | 347.1 | 3,487 | 347.1 | 0 | 12.3 |
  | Swahili (`sw`) | 407.5 | 330.3 | 153 | 317.8 | 1 | 42.6 |
  | Italian (`it`) | 357.8 | 304.5 | 1,444 | 304.5 | 0 | 11.4 |
  | Mandarin (`zh-CN`) | 254.1 | 215.2 | 2,031 | 215.2 | 0 | 1.6 |
  | Bashkir (`ba`) | 257.8 | 213.4 | 142 | 211.4 | 1 | 32.0 |
  | Russian (`ru`) | 254.4 | 203.4 | 624 | 203.4 | 0 | 13.4 |
  | Georgian (`ka`) | 251.0 | 193.2 | 1,292 | 154.6 | 1 | 68.6 |
  | Cantonese (`yue`) | 204.1 | 182.3 | 111 | 182.3 | 0 | 9.6 |
  | Tamil (`ta`) | 234.5 | 182.1 | 29 | 120.2 | 1 | 91.9 |
  | Abkhaz (`ab`) | 204.9 | 157.4 | 376 | 157.4 | 0 | 2.8 |
  | Portuguese (`pt`) | 184.1 | 133.7 | 567 | 133.7 | 0 | 15.8 |
  | Polish (`pl`) | 175.6 | 122.9 | 299 | 122.9 | 0 | 12.3 |
  | Thai (`th`) | 178.9 | 117.2 | 294 | 114.2 | 1 | 33.0 |
  | Welsh (`cy`) | 120.9 | 89.9 | 505 | 89.9 | 0 | 5.2 |
  | Turkish (`tr`) | 129.2 | 86.5 | 57 | 86.5 | 0 | 23.6 |
  | Kazakh (`kk`) | 2.5 | 0.9 | 6 | 0.9 | 0 | 0.2 |
  | **Total, 27 locales** | **18,683.8** | **17,329.2** | **143,206** | **15,575.3** | **49** | |

- Cantonese also has `zh-HK`, which the index maps to `yue` (`commonvoice.LANGUAGE`), extracted after this table: 109.0 h validated, 82.2 h in the train pool of 728 speakers, none over 30 h, the largest 5.5 h (measured 2026-11-08 by the same rule). So Cantonese has 313.1 h validated and 264.5 h in the train pool, over 200 h either way.

## Per-source facts

Pins, URLs and checksums are in the fetchers (`training/src/openvad_train/fetch/`); licences and fetch details are also in [`data/README.md`](../../data/README.md).

### SWC

- Spoken Wikipedia Corpora 2.0, Hamburg FDR record 1875, CC BY-SA 4.0: 37.6 GB zip with English (16.2 GB), German (14.2 GB) and Dutch tars; one `audio.ogg` per article.
- Each article's audio keeps its Wikimedia Commons licence: English 1,174 of 1,314 articles CC BY-SA (171.9 h aligned words), 66 public domain or CC BY, 47 GFDL-only, 27 without a recognised licence; German 948 of 1,010 CC BY-SA (246.0 h), 46 public domain or CC BY, 4 GFDL-only, 12 unknown.
- Alignments are word timings (`<n start end>` in ms) from Sphinx-4; German acoustic models were bootstrapped from earlier models (Baumann 2010) and Voxforge, the English model's training data is not documented.
- Only aligned words are speech; a gap between two aligned words with no unaligned token between them is non-speech (about 32 h English, 46 h German); everything else is unknown, not non-speech.
- Aligned words sum to 516.9 h: English 184.1 h, German 254.2 h, Dutch 78.6 h (measured 2026-11-06, summing word timings per article).
- Reader names come from each article's `info.json`, with wiki-link forms and case merged by hand-written rules, so about 900 is approximate; 25.6 h has no reader name and is not capped.
- Two readers pass 30 h: Blik (German, 76.4 h) and Jcb (Dutch, 66.1 h, 84% of the Dutch hours). The cap leaves 434.4 h: English 184.1 h, German 207.8 h, Dutch 42.5 h. The train figure assumes 80% of that and neither reader in dev or test; the split is not drawn yet.

### MLS

- OpenSLR 94, CC BY 4.0, read LibriVox audiobooks in 8 languages, 16 kHz (downsampled from 48 kHz).
- Train hours and speakers: English 44,659.7 h, 5,490; German 1,966.5 h, 176; Dutch 1,554.2 h, 40; French 1,076.6 h, 142; Spanish 917.7 h, 86; Italian 247.4 h, 65; Portuguese 161.0 h, 42; Polish 103.7 h, 11. Dev and test are about 2 to 16 h each per language, with 2 to 21 speakers per gender.
- Splits are speaker-disjoint, and each chapter appears in one split only. The English train set shares speakers and books with LibriSpeech train; its dev and test sets are independent of LibriSpeech's.
- Segments are 10 to 20 s, cut at silences found by an acoustic model; transcripts were matched from the book texts, no alignments are shipped.
- Archives (FLAC / Opus): English 2.4 TiB / 651 GiB, German 115 / 29 GiB (German FLAC measured: 123 GB), Dutch 86 / 23 GiB, French 61 / 16 GiB, Spanish 50 / 14 GiB, Italian 15 / 3.8 GiB, Portuguese 9.3 / 2.5 GiB, Polish 6.2 / 1.6 GiB. The Opus set averages about 35 kbit/s (651 GiB over the English hours). The seven FLAC archives used total ≈ 365 GB.

### VoxPopuli

- `facebookresearch/voxpopuli`: European Parliament plenary speech, semi-spontaneous, 16 kHz mono Ogg Vorbis segments cut from whole sessions.
- Licence: data CC0 (the raw audio also falls under the European Parliament's legal notice). The README puts the code under CC BY-NC 4.0 while the source headers say MIT; the code is not vendored, and the little we need (download the session tars, read the annotation table, cut segments) is reimplemented.
- Transcribed part: 16 languages; the fetcher takes 13. Left out: Slovak (35 h), Slovene (10 h) and Lithuanian (2 h).
- Transcribed part (the paper's figures): 1,791 h in 16 languages, for example English 543 h and 1,313 speakers, German 282 h and 531, French 211 h and 534, Spanish 166 h and 305, Polish 111 h and 282, Italian 91 h and 306, Dutch 53 h and 221. The annotation tables give somewhat different hours without the `invalid` rows (French 222.8 h, English 546.5 h).
- The annotation table (`annotations/asr/asr_<lang>.tsv.gz`) gives session, speaker ID, start and end, original and normalised text, a decoded hypothesis with CER and WER, VAD intervals and a split (`train`, `dev`, `test`, `invalid`). `invalid` rows are segments whose text does not match the audio (English 724.7 h, German 201.3 h).
- The official splits are not speaker-disjoint: English test shares 109 of its 286 speakers with train, German 72 of 123, French 77 of 144, Spanish 55 of 96.
- The source audio of the transcribed part is one set of 12 yearly tars (`audios/original_2009.tar` to `original_2020.tar`), 77.3 GB for all 16 languages.
- The 384k h unlabelled set has no labels.

### AMI

- CC BY 4.0 (Edinburgh); the OpenSLR 16 page states CC BY-NC-SA 2.0. 100 h of English meetings, 171 meetings, spontaneous speech with overlap; individual headset (IHM), headset mix and distant microphone arrays, 16 kHz.
- Manual transcripts with word timings (`ami_public_manual_1.6.2.zip`, 22.9 MB).
- The full-corpus-ASR partition is speaker-disjoint (checked against `corpusResources/meetings.xml`): train 136 meetings, 152 speakers, 80.6 h; dev 18 meetings, 21 speakers, 9.7 h; test 16 meetings, 16 speakers, 9.1 h.
- Meetings with five participants also have `Headset-4`.
- Each audio stream is about 11.5 GB for the whole corpus (16 kHz, 16-bit, measured on `ES2002a`); IHM has one stream per participant, about 46 GB for four.
- In an IHM channel the wearer is the dominant speaker and the others are crosstalk.
- The adapter (`dataset/sources/ami.py`, measured on the share) gives 678 headset clips of 189 participants: train 543 clips and 323.1 h, dev 72 and 38.7 h, test 63 and 35.4 h; the words of a wearer's `words.xml` are 93.1 h of speech. Of 1,147,783 `<w>` elements 4 have no times and 11 end before they start; they are dropped. Vocal sounds, disfluency markers and gaps are other elements.
- Cut by `split_over` the 397 h become 61,562 samples with median 24.8 s. A sample is never cut inside speech, so 104 of them (0.17%) pass 30 s, at most 40.2 s, where a word crosses the limit or a last piece under 1 s joins the one before. 13,510 samples (94.3 h of the 397.2 h) hold no speech, because a participant is quiet for long stretches; they are kept as non-speech examples, and capping them is an owner decision.
- Every AMI headset WAV is far above `_LONG_FILE_S` (60 s), so a first read of one also pays the whole-file integrity decode of `decode_16k` once per process, as VoxPopuli does (see "Reading as-is audio over the share").

### CallHome

- TalkBank mirror on Hugging Face (`talkbank/callhome`, gated), CC BY-NC-SA 4.0; TalkBank itself is CC BY-NC-SA 3.0 and excludes commercial AI products. The paid Linguistic Data Consortium edition is LDC2026S06.
- Calls (from the dataset card; hours measured to the last turn end): English 140 calls, 20.3 h; German 120, 18.4 h; Japanese 120, 18.7 h; Spanish 140, 21.9 h; Mandarin 140, 20.3 h; 660 calls, 99.6 h, 11.2 GB of parquet. The union of the turns is 89.1 h of speech; the calls carry 1,556 speaker labels.
- The audio is mono 16-bit WAV at 16 kHz, cut to the transcribed span of each call. The original calls are 8 kHz telephone recordings; the mirror documents neither the upsampling nor the mix of the two sides into one channel. The files are used unchanged, with `band = narrow_native`, so no train-time narrowband step is applied ([0021](../project/adr/0021-telephone-channel-simulation.md)).
- Labels: start and end time and speaker letter (`A`, `B`, sometimes `A1`, `B1`, `B2`) per turn; no transcripts and no speaker IDs across calls. Turns of the two sides overlap, and pauses inside a turn count as speech.
- A shard has no call IDs and no audio path, only the row; a call is `<language>/<shard stem>/<row>`. The last turn ends where the audio ends (checked on 3 calls, to 1 ms), so the clip is as long as its last turn.
- The adapter (`dataset/sources/callhome.py`, measured on the share) splits the 660 calls by the rank of their ID: 528 train (80.0 h), 66 validation (10.2 h), 66 test (9.5 h). The ±0.25 s around every turn boundary masks 45.5 h of the 99.6 h, because turns are dense; the speech union is 89.2 h. Cut by `split_over` the calls become 16,110 samples with median 22.4 s, of which 297 (1.84%) pass 30 s (at most 132.4 s) because overlapping turns leave no gap to cut at.
- Reading one row means decoding the call out of its Parquet row group: about 6 s each for 300 to 600 s calls on the M3 Pro over SMB.

### FLEURS

- Hugging Face `google/fleurs`, CC BY 4.0, not gated: read Wikipedia sentences (FLoRes) in 102 languages, 16 kHz WAV.
- The 42 fetched languages hold 397.6 h of train, 49.3 h of dev and 115.8 h of test, 129.4 GB on disk: 13.4 h per language on average, and from 5.1 h (Georgian) to 13.6 h (Hausa) of train.
- Examples: German train 9.0 h, dev 1.3 h, test 3.2 h; English 7.5, 1.1 and 1.8 h. Per language about 2.4 GB, of which the test archive is about 0.6 GB; about 248 GB for all languages.
- Train is speaker-disjoint from dev and test according to the FLEURS paper; the TSVs carry gender but no speaker IDs, so this cannot be checked.

### Speech Commands

- v0.02 (CC BY 4.0): 105,829 one-second words of 35 keywords from 2,618 speakers; the official `validation_list.txt` and `testing_list.txt` split by speaker hash. Labels are per clip only.

### VOiCES

- CC BY 4.0, no credentials needed, S3 bucket `lab41openaudiocorpus`.
- Devkit 27.5 GiB (29.5 GB), a subset of the release (recorded as 417.5 GiB and as 448.8 GB, which is 418.0 GiB).
- The release re-records 3,903 utterances (≈ 15 h) 256 times, the devkit 600 of them 32 times.
- The devkit holds ≈ 84 h of speech, plus ≈ 235 h of distractor-only room noise.
- Its speech is re-recorded LibriSpeech "clean". Of its 300 speakers, 12 are from `dev-clean`, 64 from `train-clean-100`, 224 from `train-clean-360` and none from `test-clean` (checked 2026-11-03).
- The devkit ships the 600 source recordings (`source-16k/{train,test}/sp<speaker>/Lab41-SRI-VOiCES-src-*.wav`), 16 kHz and as long as their re-recordings, so the delay is measured on the files and no speaker is left uncovered. `references/{train,test}_index.csv` list the 19,200 re-recordings (12,800 and 6,400); `references/time_values.csv` is the release's table of lags for all 998,400 files, with the opposite sign to ours. The lag measured by cross-correlation agreed with it to the sample on 30 of 30 random devkit recordings (up to 972 samples, 61 ms). The search window is ±0.5 s, because 954 of the 18,432 recordings lag by 0.25 to 0.5 s (a window of ±0.25 s put the peak at an arbitrary lag for them, and for 2 near 7.9 s).
- The adapter (`dataset/sources/voices.py`, measured on the share) gives 18,401 clips: train 15,526, validation 1,405, test 1,470. Of the 19,200 recordings, 768 of the 12 `dev-clean` speakers are dropped, and 31 more because the measured lag differs from the devkit table by over 16 samples (`delay_disagrees_with_devkit`; none is missing from the table, `no_devkit_delay` is 0); 952 kept recordings lag by 0.25 to 0.5 s. Before the table check the clips were 288 speakers: 243 train (68.7 h; the 64 `train-clean-100` speakers and 179 of the 224 others), 22 validation (6.2 h) and 23 test (6.4 h). Measuring all delays took 549 s for 18,432 recordings (each recording and, once per 32, its source are read); `index` measures them in `clips` and reuses them in the same run, `plan` measures again for the clips that remain.

### Noise: DEMAND, MUSAN, WHAM!

- DEMAND (Zenodo 1227121): 16 environments; the record metadata says CC BY 4.0, its description CC BY-SA 3.0; the stricter one is recorded. SCAFE exists only at 48 kHz, so Street has two scenes (SPSQUARE, STRAFFIC) where LibriVAD has three.
  - The nine noise categories take DEMAND scenes by first letter (D, N, O, P, S, T) for Domestic, Nature, Office, Public, Street and Transport, with one random channel per draw. Channels 1 to 12 are train noise, 13 and 14 validation and 15 and 16 test ([0023](../project/adr/0023-dataset-composition.md)).
- MUSAN (OpenSLR 17, every subfolder LICENSE read): no NC files.
  - 1,970 of 2,016 files are public domain, CC BY or CC BY-SA (music 40.1 h, noise 6.2 h, speech 60.5 h). Their music is 20.8 h public domain or CC BY and 19.3 h CC BY-SA; 12.8 h of it has vocals.
  - The other 46 are music files (2.5 h): 32 CC BY-ND 3.0 and 14 with no licence statement (8 in `fma-western-art`, 6 in `jamendo`, 2 of them absent from the LICENSE file).
- MUSAN speech: LibriVox readings tagged per file in 15 languages (English 76 files, German 16, Spanish 11, French 11, Russian 9, Portuguese 9, Dutch 9, Italian 8, Chinese 7, Hebrew 4, Tagalog 3, Japanese 3, Polish 2, Latin 1, Hungarian 1); no word labels, and not used: MUSAN contributes noise only, and its music is not used either ([0023](../project/adr/0023-dataset-composition.md)). Its 6.2 h of noise are sorted into the nine noise categories by keyword.
- WHAM! noise (`wham_noise.zip`, CC BY-NC 4.0 on the official page `wham.whisper.ai`):
  - Noise only, no WSJ0 mixtures. Recorded in late 2018 in restaurants, cafes, bars and parks in the San Francisco Bay Area with a binaural microphone pair; segments with intelligible speech were removed.
  - 32-bit float WAV, 2 channels, 16 kHz. Splits (from the zip directory, 128,000 bytes per second): `tr` 20,000 files, 58.0 h; `cv` 5,000 files, 14.7 h; `tt` 3,000 files, 9.0 h; 81.7 h in total, 37.6 GB unpacked.
  - `metadata/noise_meta_*.csv` give the recording, location and day of each clip. The splits share no recording and no location (`tr` 25 locations, `cv` 11, `tt` 8), so `cv` and `tt` are unseen locations for a model trained on `tr`.
  - LibriVAD's City noise is cut from WHAM! by split (`Scripts/create_city_noise.py`): 3 h of `tr` for its train, 30 min of `cv` for validation, 30 min of `tt` for test.
  - OpenVAD's City category takes one random WHAM! channel per draw, `tr` for train and `cv` for validation, and (proposed, as in LibriVAD) `tt` for test; `tt` never trains.
- Distinct ambient noise for training is ≈ 65 h: ≈ 7.5 h from DEMAND and MUSAN noise, 58.0 h from WHAM! `tr`.

### LibriSpeech and LibriVAD

- The LibriSpeech alignments of Zenodo 2619474, and the copy that LibriVAD appears to ship, are in [`forced-alignment.md`](forced-alignment.md#prior-use-for-vad-and-segment-labels).
- LibriVAD's speech is `train-clean-100`; its noise comes from DEMAND, WHAM! `tr` and its own babble and speech-shaped noise, and includes WHAM! (CC BY-NC 4.0).
- `medium` takes every 10th utterance of `train-clean-100` and mixes each one 54 times, so its 546 h NonConcat and 680 h Concat hold about 10 h of unique speech, not the 101 h of `train-clean-100`; each utterance appears 54 times by design.
- Splits: `medium` train is used as generated. `medium` test (from `test-clean`) is used as generated, in the test split and as the benchmark of [0020](../project/adr/0020-evaluation-benchmarks.md).
- `medium` val (from `dev-clean`) does not exist at the pinned revision, because `create_LibriSpeech_wav.py` converts only two splits; `just fetch librivad-val` generates it into `datasets/raw/librivad-val/` with its own manifest, because `create_LibriVAD.py` and `create_LibriSpeechConcat.py` already handle `val`, and `Files/Noises/*_val.wav` and the `dev-clean` alignments exist. It holds about 27.4 h NonConcat and 35.2 h Concat, validation only ([0024](../project/adr/0024-training-data-licence-policy.md)).
- Labels (`Files/Labels/<dataset>/<split>/<speaker>/<chapter>/<stem>.npy`) are one `int16` per sample, 1 speech and 0 non-speech. The 54 mixtures of an utterance share one label, which is as long as the wav for most (checked on 4 mixtures) but not for all. 36 plain `LibriSpeech` utterances (1,836 train and 108 test mixtures, 2 utterances of `test-clean`) have a wav exactly one sample longer than the label: the same signal, so the adapter keeps them and repeats the last label value once (`label_one_sample_short` in the index manifest). Every other difference is dropped and counted (`label_length_mismatch`): 3,834 mixtures of `LibriSpeechConcat` (3,456 train, 378 test, 4.6% of the concatenated mixtures), among them the pairs of two chapters or speakers and some of one chapter. A concatenated pair `A_+_B` has its label in the folder of `B`, not of `A` (`1995-1836-0014_+_1837-0000` is under `1995/1837`, `1580-141084-0050_+_1995-1826-0000` under `1995/1826`), so upstream `match_labels.py`, which reads the folders from the first part, is wrong for pairs of different chapters or speakers; the adapter derives the folder from `B`, and all 252,396 mixtures of the medium train and test find a label file that way.
- The adapter (`dataset/sources/librivad.py`, measured on the share) gives 248,562 clips: train 227,718 and test 20,844, after dropping the 3,834 above (train 3,456, test 378); before the drop it was 252,396 clips of 291 speakers. The first measurement, which also dropped the one-sample cases, gave 246,618 clips (train 225,882, 1,186.6 h; test 20,736, 64.4 h); the counts here add the 1,944 clips back from the 36 utterances and were not remeasured. The validation layout is read from `datasets/raw/librivad-val/{generated,labels}` and was built against a synthetic tree, because `librivad-val` is not generated yet. Reading the 252,396 wav headers took 42 min while another job read the share (one 4 KB read each, 14 ms).
- The label tree holds a file for every utterance of a split (2,620 for `test-clean`), not only for the medium subset, so listing it does not give the clips; the adapter lists the wavs.
- A LibriVAD-like set could be rebuilt by our own code from DEMAND, WHAM! and babble from our own speech sources, without LibriVAD's scripts and `Noises.zip`.

### Room impulse responses

- BUT ReverbDB: CC BY 4.0, about 8.7 GB of real room impulse responses.
- OpenSLR 28 (RIRS_NOISES): Apache-2.0, about 1.3 GB.
- Both are used for the room step of [0021](../project/adr/0021-telephone-channel-simulation.md) (p = 0.5), with rooms split 80/10/10 by hash and each response peak-aligned to zero. `just fetch rirs` is to fetch both; neither is fetched yet, and the sizes are from the dataset construction spec, not from a manifest.

### AVA-Speech and Picovoice

- AVA-Speech: film audio in many languages; 151 of the 160 labelled videos are on the S3 mirror; the other 9 are listed in `missing.txt` and skipped.
- Picovoice voice-activity-benchmark (Apache-2.0) mixes LibriSpeech `test-clean` with 10 DEMAND environments at 0 dB, so these recordings are seen noise for a model trained on DEMAND.

### Fearless Steps

- Challenge Phase 4: CC BY 4.0 for the 100 h 8 kHz challenge corpus, its metadata and scripts by CRSS-UTDallas; the audio is English from the Apollo missions and itself falls under the NASA media usage guidelines.
- Access needs registration and acceptance of the challenge terms.
- Speech activity labels are human, as start and end times, scored with a collar (0.5 s in Phase 1 and 2, 0.25 s in Phase 3).

### Simsamu

- Hugging Face `medkit/simsamu`, MIT for the whole repository, audio and annotations. 61 simulated French emergency calls by 14 intern speakers, 3.22 h annotated, 2.61 h speech; mono AAC at 8 kHz, about 38 kbit/s (the README says 128 kbit/s).
- The caller's voice runs through the telephone network and a phone loudspeaker into the microphone; the regulator speaks into the microphone directly, so only the caller side is a telephone channel.
- The RTTM segments are speaker turns (one per SRT subtitle), so pauses inside a turn count as speech, as in CallHome; they never overlap. The gaps between turns have a median of 0.56 s, 31% are under 0.25 s. The word times in the JSON files look evenly spread, not aligned, and are not used.
- The README does not say whether the speakers consented to publication.

## Constructed audio: size and speed

Measured on the first real slice run (`openvad-slice`: 500 SWC German samples planned, 498 built, 2.09 h), on the macOS M3 Pro (6 performance and 6 efficiency cores) reading the share over SMB, ffmpeg 8.1.3, with the construct and codec code of commit `59515d6`.
These numbers size shards and plan runs; they are not the headline real-time factor, which comes from the reference CPU.

- Size: 135.1 MB of uncompressed tars for 2.086 h, which is 64.8 MB per hour (64.6 MB per hour of FLAC members).
  Per shard: train 64.9, validation 65.0, test 63.4 MB per hour. Half the samples go through G.722 and 84% have DEMAND noise, which is why this is above a clean-speech FLAC figure.
  `flac_mb_per_hour` in `BuildConfig` is set to 64.8.
- Speed: 7 min 8 s wall for the three shards (so three of the 11 workers ran), 617 s user plus 636 s system time, 2.5 CPU seconds per sample: **0.40 samples per second per core**.
  Seen per worker, the train shard built 398 samples in 428 s (0.93 samples per second per worker, each worker using about two cores with its ffmpeg and SMB reads).
- Integrity decode: every source file over 60 s is decoded once per process by `_verify_intact` before it is read. The slice's 316 distinct source files hold 162 h of Ogg (5.6 GB), and decoding all of them took 382 s on a cold cache, 1.21 s per file or about 1,500 times real time.
  That is about 30% of the construct CPU time at this scale. At full scale it is one decode per long file per worker, so a file that several workers read is decoded several times.
- One SWC Ogg (`german/Nike_des_Paionios/audio.ogg`) fails the integrity decode with a CRC mismatch; its two sampled chunks were left out and logged, as designed.

## Reading as-is audio over the share

Measured on 2026-11-08 on the macOS M3 Pro (12 cores), Python 3.12.15, reading the share over SMB (`/Volumes/datasets/openvad`) through `load_audio`, one process and one read at a time, nothing copied. Rows are random with seed 0, built with `draw_sample` from the adapters' clips, so a row is a whole clip. These are the cold reads a training worker makes on first touch, not the headline real-time factor.

| Source | Reads | Distinct files | Reads per second | Median, 95th percentile, maximum |
| --- | --- | --- | --- | --- |
| Speech Commands (1 s WAV) | 1,000 | 1,000 | 6.06 | 143 ms, 280 ms, 674 ms |
| VoxPopuli (median 9.2 s of a session Ogg), integrity decode on first read | 691 of 1,000 (stopped at the 25 min budget) | 578 | 0.46 | 1.56 s, 7.8 s, 12.9 s |
| VoxPopuli, `_verify_intact` replaced by a no-op (seek only), 200 rows with seed 1 | 200 | 184 | 3.52 | 265 ms, 541 ms, 721 ms |

- A Speech Commands read spawns one ffmpeg per clip (`decode_16k` decodes a file of up to 60 s whole).
- A VoxPopuli session file is hours long, so `decode_16k` first decodes all of it once per process (`_verify_intact`, 4096 paths cached) and then seeks with soundfile. That first decode is what the first row costs: 0.46 reads per second against 3.52 with it off, about 7.5 times (two different random samples). A training worker pays it once per session file it touches, so with several workers the same file is decoded several times, and the 598,179 clips lie in 4,611 session files.
- Seeking alone is fast enough (about 0.27 s per row), so the open question for training is the integrity decode, not the seeks. Doing it once in `verify`, or caching the result on the share, would remove it from the reads; neither is built, and no audio is copied (owner's decision).
- A CallHome read decodes the whole call out of its Parquet row group, about 6 s for a 300 to 600 s call (3 calls timed), and a 30 s sample of it repeats that. Training on CallHome as it is needs the calls cached locally, which the owner has to decide.

## Scale references

- Granary v2 labels 463,062 h of speech, about 165,300 h of it flagged high quality, mostly unlabelled VoxPopuli (380,711 h) and LibriLight (49,334 h), with Whisper-style pseudo-transcripts and no word timings.
- LibriLight, which Granary draws on: audio from LibriVox, public domain; code MIT.
- An ASR corpus needs that volume for transcription; a VAD with a size budget of about 1 MB needs frame labels and diversity more than volume.
- Granary's high-quality pool would only matter as a filter or transcript source for unlabelled audio.
- Other corpora seen in Granary: People's Speech `clean` (EN), Europarl-ASR (EN, 1,023 h), VCTK (EN), FalAR (PT, 4,715 h, CC BY 4.0), Golos and Dusha (RU, custom licences), National Speech Corpus (Singapore EN).

## Sources

- Common Voice 27.0 statistics: `common-voice/cv-dataset`, `datasets/scripted-speech/cv-corpus-27.0-2026-09-11.json`, commit `65ca2460d5e686ea03902a932bbca09b322dc4b0`.
- Granary v2: <https://huggingface.co/datasets/nvidia/Granary-v2>.
- Language families: <https://en.wikipedia.org/wiki/List_of_language_families>.
- MFA models: <https://github.com/MontrealCorpusTools/mfa-models/releases>.
- FLEURS: <https://huggingface.co/datasets/google/fleurs>.
