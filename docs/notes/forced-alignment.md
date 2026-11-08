# Forced alignment

A comparison of the Montreal Forced Aligner (MFA 3.x with its pretrained models) and the MMS forced aligner in torchaudio (`MMS_FA`) as labelling tools for OpenVAD.
State of 2026-11-04.

The labels we need: word-level speech intervals from forced alignment, turned into 10 ms speech/non-speech frames, where a gap between two words shorter than G = 200 ms (owner, 2026-11-06) counts as speech and a longer one as non-speech ([0022](../project/adr/0022-frame-labels.md)).
The data: Common Voice 27.0, Multilingual LibriSpeech, VoxPopuli, FLEURS and Speech Commands (train, dev and test) in about 53 languages, counting Common Voice's 53 locales as 51 languages plus Bosnian and Croatian from FLEURS ([0023](../project/adr/0023-dataset-composition.md)).
Every section before [Recommendation](#recommendation) states sourced facts only; the recommendation records what [0025](../project/adr/0025-alignment-labelling.md) proposes (pretrained MFA models) and the alternative it did not adopt (`MMS_FA`).
Anything not checked against a primary source is marked "not verified".

## Summary

| Aspect | MFA 3.x | `MMS_FA` (torchaudio) |
| :-- | :-- | :-- |
| Method | Gaussian mixture model with hidden Markov model (Kaldi), phones from a pronunciation dictionary | wav2vec 2.0 CTC model, characters of uroman-romanised text |
| Per language | acoustic model, dictionary, G2P for out-of-vocabulary words | none; text romanised with uroman and normalised to `a` to `z` and `'` |
| Output | word and phone intervals | character spans, grouped into words by the caller |
| Frame step | 10 ms | 20 ms |
| Word onset error, English (MFA 2026 paper) | 19.9 to 25.4 ms mean | 43.1 to 49.5 ms mean |
| Pauses | optional silence phone `sil` between words | CTC blank, shared with "repeat the previous token" |
| Failure signal | beam failure, per-utterance log-likelihood and other metrics in `alignment_analysis.csv` | per-frame probability of the aligned token |
| Hardware | CPU, multiprocessing | CPU or CUDA |
| Code licence | MIT | BSD-2-Clause (torchaudio) |
| Model licence | CC BY 4.0 (`*_mfa`), CC0 (`*_cv`) | CC BY-NC 4.0 |
| Status | v3.4.3, 2026-10-08; 4.0 changes the `align` signature | `forced_align` and `MMS_FA` still in torchaudio 2.11.1 (2026-10-28) |

## Alignment accuracy

All published boundary evaluations found are on English corpora (TIMIT, named after Texas Instruments and Massachusetts Institute of Technology, and Buckeye), except one preprint with German, Dutch and Hebrew.
None is on Common Voice, audiobooks, parliament speech or noisy audio.

### Word boundaries

MFA 2026 paper (McAuliffe et al., Interspeech 2026, Table 4): word onsets, mean absolute error in ms and share of onsets within a tolerance.

| Aligner | TIMIT mean | ≤ 10 ms | ≤ 25 ms | ≤ 50 ms | ≤ 100 ms | Buckeye mean | ≤ 10 ms | ≤ 25 ms | ≤ 50 ms | ≤ 100 ms |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| MFA ARPA 3.0 (`english_us_arpa`) | 19.93 | 44.99% | 66.50% | 91.61% | 98.38% | 21.75 | 48.76% | 70.16% | 91.35% | 97.29% |
| MFA Global 3.0 (`english_mfa`) | 22.33 | 42.33% | 61.98% | 89.53% | 97.73% | 25.35 | 47.27% | 67.77% | 89.30% | 95.51% |
| MFA ARPA 1.0 | 28.18 | 35.07% | 56.95% | 84.03% | 95.76% | 27.22 | 39.30% | 61.99% | 87.91% | 95.65% |
| MMS | 43.06 | 13.05% | 26.52% | 63.73% | 95.91% | 49.54 | 9.58% | 20.27% | 61.85% | 92.05% |

- The paper evaluates word onsets ("beginning timestamps"), not ends.
- Spontaneous-speech utterances were cut at 300 ms silences and padded with 200 ms of silence on either side; utterances of three words or fewer were dropped.
- The paper's scripts run `MMS_FA` with the reference transcript and `with_star=False` (`alignment/word_align_w2v2.py` in `mfa-interspeech2026`); that this script produced the `MMS*` row is my reading of the repository, not stated in the paper.
- The paper concludes that MFA 3.0 "substantially outperforms all three neural ASR-based aligners on both datasets", with the largest gap at 10 and 25 ms.

Rousso et al. 2024 ("Tradition or Innovation", Interspeech 2024): word end timestamps, share within a tolerance.

| Aligner | TIMIT ≤ 10 | ≤ 25 | ≤ 50 | ≤ 100 | Buckeye ≤ 10 | ≤ 25 | ≤ 50 | ≤ 100 |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| MFA (982 h LibriSpeech model, ARPABET) | 41.6% | 72.8% | 89.4% | 97.4% | 39.8% | 69.9% | 84.9% | 91.8% |
| MMS | 18.6% | 43.5% | 75.7% | 94.7% | 25.0% | 52.7% | 75.0% | 87.9% |

- Mean / median word end error, TIMIT: MFA 21.9 / 12.5 ms, MMS 68.5 / 29.3 ms.
- Buckeye: MFA mean 976.5 ms (median 13.6 ms), which the authors attribute to drift on input utterances "as long as several minutes"; with errors over 500 ms removed, MFA 27.8 / 12.9 ms and MMS 41.0 / 22.2 ms.
- F1 within 20 ms: TIMIT MFA 65.7, MMS 35.4; Buckeye (errors over 500 ms removed) MFA 65.4, MMS 46.3.
- Only words that MMS and WhisperX recognised correctly were compared (29,057 of 39,834 TIMIT words for MMS), which suggests MMS ASR output rather than `MMS_FA` on the reference; the paper does not name `MMS_FA` (not verified which was used).
- The MFA model is named only as "trained on 982 hours of LibriSpeech" with an ARPABET lexicon; that it is `english_us_arpa` is not verified.

Huang et al. 2024 ("Less peaky and more accurate CTC forced alignment by label priors", ICASSP 2024 per its arXiv comment; co-authors include the torchaudio and MMS authors), Table 1.

| Aligner | Buckeye WBE | Buckeye WDUR | TIMIT WBE | TIMIT WDUR |
| :-- | --: | --: | --: | --: |
| MMS (torchaudio CTC alignment tutorial) | 53 ms | 181 ms | 37 ms | 242 ms |
| MFA `english_mfa` | 41 ms | 251 ms | 23 ms | 313 ms |
| MFA `english_mfa`, fine-tuned on Buckeye | 36 ms | 250 ms | 22 ms | 314 ms |
| Ground truth | 0 | 241 ms | 0 | 305 ms |

- WBE is the mean of the absolute onset and offset errors per word; WDUR is the mean predicted word duration.
- MMS words come out about 60 ms shorter on average than the reference on both corpora; MFA words about 10 ms longer.
- For a standard CTC model of their own, onset errors exceed offset errors (word: 63 vs 54 ms), which the authors read as CTC tending "to delay token predictions".

mfa-models "English alignment benchmarks" page (MFA 3.1.0), mean word boundary error.

| Aligner | Buckeye | TIMIT |
| :-- | --: | --: |
| MFA ARPA 1.0 | 28.8 ms | 25.2 ms |
| MFA 3.0 (`english_mfa` with US MFA dictionary) | 26.9 ms | 21.0 ms |
| Wav2Vec2 (`MMS_FA`) | 47.6 ms | 37.0 ms |
| WhisperX (`WAV2VEC2_ASR_BASE_960H`) | 58.7 ms | 46.2 ms |
| NeMo Forced Aligner | 95.5 ms | 77.1 ms |

- The page says ARPA 3.0 "shows the best performance across both corpora", but its table has no ARPA 3.0 row (number not verified).
- The page warns against basing decisions on TIMIT: slow, phonetically balanced read sentences, 5 h.

MFA 2017 paper (McAuliffe et al., Interspeech 2017), MFA 1.0 with the LibriSpeech model.

- Buckeye word boundaries: mean 24.1 ms, median 15.8 ms; 33%, 68%, 88% and 97% within 10, 25, 50 and 100 ms.
- The paper compares the 24 ms mean with 26 ms inter-transcriber reliability, and notes 68% within 25 ms against 90% inter-transcriber agreement at 26 ms.
- 2 to 5% of boundaries differ by at least 100 ms.

Weber et al. 2026 (arXiv 2606.10675, preprint; Interspeech 2026 acceptance not verified), word boundary accuracy outside English.

| Corpus | Aligner | ≤ 10 ms | ≤ 25 ms | ≤ 50 ms | ≤ 100 ms |
| :-- | :-- | --: | --: | --: | --: |
| German PHONDAT | MFA | 29.9% | 65.4% | 82.1% | 94.3% |
| German PHONDAT | MMS | 21.8% | 44.3% | 74.9% | 91.8% |
| Dutch IFA | MFA | 4.7% | 7.3% | 11.6% | 19.0% |
| Dutch IFA | MMS | 16.0% | 37.9% | 62.9% | 76.6% |
| Hebrew | MMS (no MFA model) | 14.3% | 41.3% | 76.5% | 94.7% |

- The paper does not say which MFA models and dictionaries it used for German and Dutch, nor why MFA fails on Dutch (not verified).

### Phone boundaries

- MFA 2017: Buckeye phone boundaries mean 17.0 ms, median 11.2 ms; 41%, 77%, 93%, 98% within 10, 25, 50, 100 ms; the 17 ms equals the reported inter-transcriber figure.
- mfa-models benchmarks (MFA v2.0.0rc8, "alignment score", mean phone boundary error): `english_us_arpa` 2.0a 19.9 ms Buckeye, 17.0 ms TIMIT; `english_mfa` 2.0a 19.4 ms Buckeye, 17.6 ms TIMIT.
- MFA 2026 paper: mean phone boundary error drops from 16.4 to 12.1 ms on TIMIT and from 17.6 to 13.9 ms on Buckeye (ARPA, 1.0 to 3.0); Korean 20.7 to 14.8 ms; the abstract reports mean boundary errors below 15 ms on all four datasets (TIMIT, Buckeye, CSJ, Seoul Corpus).
- `MMS_FA` has no phone output (its tokens are romanised characters), so no phone boundary figures exist for it.

## Language coverage and what each needs

### MFA

- Alignment needs an acoustic model and a pronunciation dictionary; words not in the dictionary become `<unk>` (phone `spn`) unless a G2P model generates pronunciations (`--use_g2p` with the Hugging Face models).
- From 3.4, `MontrealCorpusTools/<name>` on Hugging Face bundles acoustic model, dictionaries and G2P models; `mfa align_hf CORPUS HF_MODEL OUTPUT` replaces the separate dictionary argument.
- Hugging Face hosts 20 such models, all `*_mfa`: Bulgarian, Czech, English (`english_mfa`, `english_us_arpa`), French, German, Hausa, Japanese, Korean, Mandarin, Polish, Portuguese, Russian, Spanish, Swahili, Swedish, Thai, Turkish, Ukrainian, Vietnamese.
- Their `acoustic/meta.json` gives MFCC features at a 10 ms frame shift, 25 ms window, 16 kHz; Polish, Swahili and German say version 3.4.3, trained October 2026; `english_mfa` says 3.3.0 (other models not checked).
- MFA 3.0 models train on large open corpora including Common Voice and Multilingual LibriSpeech (for example `polish_mfa`: 154.7 h Common Voice, 107.9 h MLS, 24.5 h GlobalPhone); noisy Common Voice enters only in the last training round (MFA 2026 paper, Table 2).
- MFA 3.0 tokenises languages without spaces: Mandarin (spacy-pkuseg), Japanese (Sudachi), Korean (Mecab-ko), Thai (pythainlp); for Japanese the G2P works on katakana and for Mandarin on pinyin.
- MFA 3.0 offers cross-language remapping (`mfa remap dictionary`) and adaptation (`mfa adapt`); remapping Korean and Japanese onto the Global English model and adapting it came close to the dedicated Korean model (Seoul Corpus mean phone error 21.8 to 15.9 ms).

Models checked per candidate language (`MontrealCorpusTools/mfa-models` releases; these are release versions, while the Hugging Face `meta.json` above gives other version strings):

| Language | Acoustic model | Trained on | Dictionary | G2P |
| :-- | :-- | :-- | :-- | :-- |
| English | `english_mfa` 3.1.0, CC BY 4.0 | Common Voice EN, LibriSpeech, and other corpora, some NC | `english_mfa` 3.1.0, CC BY 4.0 | `english_us_mfa` 3.0.0 |
| German | `german_mfa` 3.0.0, CC BY 4.0 | Common Voice DE (1,296 h), GlobalPhone DE | `german_mfa` 3.0.0, CC BY 4.0 | `german_mfa` 3.0.0, CC BY 4.0 |
| French | `french_mfa` 3.0.0, CC BY 4.0 | Common Voice FR, GlobalPhone FR, African-accented French | `french_mfa` 3.0.0, CC BY 4.0 | `french_mfa` 3.0.0 |
| Spanish | `spanish_mfa` 3.3.0, CC BY 4.0 | Common Voice ES, MLS ES, Google i18n, M-AILABS and others | `spanish_mfa` 3.3.0 (Spain, Latin America) | `spanish_spain_mfa`, `spanish_latin_america_mfa` 3.3.0 |
| Italian | `italian_cv` 2.0.0, CC0 | Common Voice IT (288 h) | `italian_cv` 2.0.0, CC0 | none |
| Dutch | `dutch_cv` 2.0.0, CC0 | Common Voice NL (93 h) | `dutch_cv` 2.0.0, CC0 | none |
| Polish | `polish_mfa` 2.0.0a, CC BY 4.0 | Common Voice PL, MLS PL, M-AILABS, GlobalPhone PL | `polish_mfa` 2.0.0a, CC BY 4.0 | `polish_mfa` 2.0.0a |
| Portuguese | `portuguese_mfa` 2.0.0a, CC BY 4.0 | Common Voice PT, MLS PT, GlobalPhone PT | `portuguese_mfa` 2.0.0a | `portuguese_brazil_mfa`, `portuguese_portugal_mfa` 2.0.0a |
| Japanese | `japanese_mfa` 3.0.0, CC BY 4.0 | Common Voice JA, GlobalPhone JA, JVS, TEDxJP-10K and others | `japanese_mfa` 3.0.0, CC BY 4.0 | `japanese_mfa` 2.0.1a, `japanese_katakana_mfa` 3.0.0 |
| Mandarin | `mandarin_mfa` 3.0.0, CC BY 4.0 | Common Voice zh-CN and zh-TW, AISHELL-3, THCHS-30, AI-DataTang, GlobalPhone | `mandarin_china_mfa` 3.0.0, CC BY 4.0 | `mandarin_china_mfa` 3.0.0 |

- Further languages with a CC BY 4.0 `*_mfa` acoustic model: Russian 3.1.0, Swedish, Turkish, Korean, Vietnamese and Thai 3.0.0.
- Hausa has `hausa_mfa` 3.0.0 and `hausa_cv` 2.0.0; Kazakh has only `kazakh_cv` 2.0.0 (VoxCommunis, CC0; `acoustic/kazakh/cv/v2.0.0`).
- Also in the `acoustic/` tree of mfa-models (`main`, checked 2026-11-04): `bulgarian_mfa` 3.0.0 and `bulgarian_cv`, `czech_mfa` 3.3.0 and `czech_cv`, `ukrainian_mfa` 3.0.0 and `ukrainian_cv`, `croatian_mfa` 2.0.0a, and `serbocroatian/mfa` 3.3.0. Bosnian has no directory of its own, and which of these labels Croatian, Bosnian or Serbian is not verified.
- The same tree has a legacy `acoustic/arabic.zip`, although Arabic counts as a language without an MFA model; whether it is usable is not verified.
- The model cards list corpora, not splits, so the aligners may have seen Common Voice and MLS dev and test audio.

The `*_cv` models in mfa-models are VoxCommunis models (Ahn and Chodroff 2022), all v2.0.0, "compatible MFA version v2.0.0", trained on Common Voice 7.0, CC0, with a `*_cv` dictionary from Epitran or XPF and no MFA G2P model.
The table lists the 14 whose hours and speakers were read; the same tree on `main` (checked 2026-11-04) holds 32 acoustic models of this kind, also for Armenian, Bulgarian, Chuvash, Czech, Greek, Guarani, Hausa, Kazakh, Kurmanji, Polish, Portuguese, Russian, Upper Sorbian (`upper_cv`), Swedish, Thai, Turkish, Ukrainian and Vietnamese.

| Model | Hours | Speakers | Phone set |
| :-- | --: | --: | :-- |
| `italian_cv` | 288 | 6,125 | Epitran |
| `bashkir_cv` | 247 | 835 | XPF |
| `tamil_cv` | 198 | 521 | XPF |
| `dutch_cv` | 93 | 1,352 | Epitran |
| `basque_cv` | 91 | 842 | XPF |
| `belarusian_cv` | 91 | 3,620 | XPF |
| `uyghur_cv` | 41 | 281 | Epitran |
| `kyrgyz_cv` | 37 | 206 | Epitran |
| `tatar_cv` | 28 | 187 | Epitran |
| `hungarian_cv` | 16 | 116 | XPF |
| `romanian_cv` | 11 | 192 | XPF |
| `georgian_cv` | 6 | 109 | XPF |
| `abkhaz_cv` | 2 | 28 | XPF |
| `uzbek_cv` | 0.24 | 5 | Epitran |

- The model cards say they were trained on read speech in low-noise environments and recommend a wider beam when data diverges.
- MFA 3.4.3 runs `italian_cv` 2.0.0 (made by MFA 2, `meta.yaml` instead of `meta.json`) and the newer VoxCommunis `ca_vxc_acoustic16` (made by MFA 2.2.17) without changes; see [MFA pilot](#mfa-pilot-2026-11-08) for how well.

VoxCommunis itself (`pacscilab/VoxCommunis` on Hugging Face, CC0, updated 2026-07-19) holds newer MFA acoustic models, lexicons and word- and phone-level TextGrids of the validated Common Voice clips, built from Common Voice 16 to 23.

- It has newer models for all 14 `*_cv` languages above (Common Voice 16 or 17; Tatar 20).
- It has models for 10 of the 18 languages surveyed here that have no model in mfa-models: Catalan (CV 16, no lexicon file listed), Welsh (20), Bengali (17), Estonian (20), Finnish (20), Kabyle (20), Kinyarwanda (17), Luganda (20), Cantonese (19; a toned model from 21) and Mongolian (19).
- It has none for Persian, Urdu (only Common Voice 7.0 TextGrids on the older OSF page, and a `urdu_cv` dictionary in mfa-models), Meadow Mari, Arabic, Laz, Kabardian, Adyghe and Puno Quechua.
- Which MFA version trained the newer models is not stated on the dataset card (the OSF page says MFA 2.0 for the original release).

### `MMS_FA`

- One acoustic model for all languages: wav2vec 2.0 (24 layers, 1024 dimensions, about 300M parameters, 1.18 GB download), 28 output tokens (blank, `a` to `z`, `'`) plus the star token.
- The torchaudio docstring says "Trained on 31K hours of data in 1,130 languages"; the multilingual tutorial says 23,000 hours from 1,100+ languages (the two disagree).
- The MMS paper's initial alignment model was trained on FLEURS and Common Voice 8.0 (8K h, 127 languages); the final 31K h model includes that data, so FLEURS was in its training data (which FLEURS splits is not verified).
- The transcript must be normalised by the caller: romanised with uroman, lower-cased, everything but `a` to `z`, `'` and space removed (tutorial code); languages without word boundaries (Chinese, Japanese, Korean) need word segmentation first.
- The tutorial notes that uroman left "1882" as digits, which normalisation then deleted; romanising numbers "is known to be a non-trivial task".
- uroman: Python package `uroman` 1.3.1.1 on PyPI (released 2024-06-27); the Perl version ended at 1.2.8 (2021).
- uroman's Chinese example in the MMS paper gives Mandarin syllables ("nijiaoshenmemingzi"); how it romanises Cantonese text is not verified.
- Star token: `get_model(with_star=True)` (the default) appends one output whose log-probability is 0 in every frame; the MMS paper inserts it at the start of each text and in place of digits so that untranscribed audio maps to it.
- The token is `*` in torchaudio's dictionary; the tutorials call it `<star>`, as does `ctc-forced-aligner`.
- The MMS language coverage table lists MMS ASR support for 15 of those 18 languages; Adyghe, Kabardian and Puno Quechua have language identification only, so the alignment model probably never saw them (not verified).

## Granularity, frame rate and pauses

### MFA

- Output is a TextGrid per file with word and phone tiers per speaker (MFA 2017 paper).
- Features use a 10 ms frame shift, so boundaries sit on a 10 ms grid; `--fine_tune` re-aligns each boundary at 1 ms steps, with an unclear true accuracy bound because features still use 25 ms windows.
- Phones use a variable 3-state topology with a 10 ms minimum duration (MFA docs; Rousso et al. still state 30 ms, which describes a fixed 3-state hidden Markov model).
- Silence: an optional `sil` phone may occur between words and at utterance edges; `silence_probability` (default 0.5 for untrained lexicons) and `initial_silence_probability` set the prior, and pretrained models carry per-pronunciation silence probabilities estimated in training (Kaldi `get_prons`).
- Silence uses 5 hidden Markov model states, other phones 3; from MFA 3.1 silence states are no longer clustered with speech states, a change made because it "led to gross alignment errors with less clean data".
- `--initial_silence_probability 0.0` forces utterances to start without silence; `--boost_silence` scales the silence probability.

### `MMS_FA`

- Frame rate is 20 ms: the convolutional front end strides 5 × 2 × 2 × 2 × 2 × 2 × 2 = 320 samples at 16 kHz (torchaudio source); each frame sees 25 ms of audio.
- `forced_align` returns one token per frame; `merge_tokens` turns them into character spans with an inclusive start and exclusive end on the 20 ms axis.
- There is no word-boundary token; the caller groups character spans by word, and a word runs from its first character's start to its last character's end.
- There is no silence token: blank covers both "repeat" and silence, and the tutorial notes that blanks next to a word are attributed inconsistently ("Inconsistent treatment of blank token").
- The tutorial cites the "peaky" behaviour of CTC (Zeyer et al. 2021): outputs spike for a label but do not last its duration; pretrained wav2vec 2.0 models tend to spike at the start of a label (paraphrased; the tutorial misspells "occurrences").
- Measured consequence on English: MMS word durations are about 60 ms shorter than the reference (Huang et al. 2024, table above).
- `forced_align` requires the number of frames to be at least the number of tokens plus the number of repeated tokens.

## Robustness and failure signals

### MFA

- Beam 10, retry beam 40 by default; an utterance that fails both is not aligned; the default beam is kept low "so that the fMLLR feature space transform for speaker adaptation is using quality alignments".
- Two alignment passes: speaker-independent, then speaker-adapted (fMLLR); speaker grouping therefore affects results, and `--uses_speaker_adaptation false` skips the second pass.
- MFA writes `alignment_analysis.csv` per utterance: alignment log-likelihood, speech log-likelihood (silence excluded), phone duration deviation (max absolute z-score of log duration), max running short intervals (runs of 10 ms intervals, a sign of text squeezed into too little audio), intensity deviation and SNR (3.4.0 added the last two and the short-interval column).
- The docs warn the log-likelihood is relative to other paths for the same utterance and shifts with speaker adaptation, dialect and noise.
- Out-of-vocabulary words are listed in `oovs_found.txt` and `utterance_oovs.txt`; the out-of-vocabulary phone `spn` can absorb "any amount of vocal noise or speech" (MFA 2017).
- MFA 2026: small transcript or dictionary errors "often only has local effects", but "even small deviations can sometimes lead to catastrophic misalignments".
- The troubleshooting page says MFA "is not intended to align single files, particularly if they are long, have noise in the background"; `mfa segment` can cut long files with energy-based or SpeechBrain VAD.
- MFA docs read here describe no number or abbreviation expansion (not verified that none exists).
- LibriSpeech precedent: 46 utterances of the 960 h could not be aligned (CorentinJ alignments); Lugosch reports "a small number" of failures.

### `MMS_FA`

- The star token handles missing text at the start (or wherever it is inserted); the MMS paper also replaced digits with it.
- `forced_align` returns a per-frame log-probability of the aligned token; the tutorials use the span-length-weighted mean probability per word as a score.
- The MMS paper filtered alignments with the length-normalised difference between the forced path and the greedy path, `(log P(Y_aligned|X) − log P(Y_greedy|X)) / T`, keeping scores above −0.2; torchaudio does not compute this score (the greedy path has to be decoded separately).
- The MMS paper chunked long recordings into 15 s pieces for the acoustic model and concatenated the emissions; its GPU Viterbi needs O(L) GPU memory and handles very long audio.
- `forced_align` only supports batch size 1.
- Neither tool reports words that are in the text but not spoken as such; for MFA the short-interval and duration metrics hint at them, for CTC a low token score does (my reading, not verified in either set of docs).

## Speed and hardware

- MFA runs on CPU only; Huang et al. 2024: "MFA does not support GPU".
- MFA uses multiprocessing; `-j/--num_jobs` defaults to 3.
- MFA 2017: the 1,000 h LibriSpeech corpus aligned in 80 h on 12 3.4 GHz processors.
- Huang et al. 2024: MFA aligned the 20 h Buckeye corpus in 9.5 min with 8 CPU jobs.
- McAuliffe's 2021 blog: MFA 2.0 aligned 16.5 h of Buckeye "under 15 minutes" on 12 cores.
- MFA installs from conda-forge (Kaldi, OpenFst, Pynini); a `Montreal-Forced-Aligner` 3.4.3 and a `kalpy-kaldi` package exist on PyPI, but a pip-only install does not work: `kalpy-kaldi` needs a Pynini built with conda-forge's API patch (`tools/README.md`).
- `forced_align` has CPU and CUDA implementations; the MMS paper's Figure 4 shows the GPU version scaling to 2,000 s inputs where CPU alternatives slow down (no throughput number given).
- `MMS_FA` CPU throughput is not documented (not verified).

## Licences

| Component | Licence | Source |
| :-- | :-- | :-- |
| MFA code | MIT | GitHub repository |
| `*_mfa` acoustic models, dictionaries, G2P (incl. `english_us_arpa`) | CC BY 4.0 | mfa-models `LICENSE`, model cards, Hugging Face cards |
| `*_cv` models and dictionaries | CC0 | mfa-models model cards |
| VoxCommunis (models, lexicons, TextGrids) | CC0 | Hugging Face dataset card |
| Epitran | MIT | GitHub |
| XPF | not verified | GitHub shows no licence |
| torchaudio code | BSD-2-Clause | GitHub |
| `MMS_FA` weights | CC BY-NC 4.0 | torchaudio docstring; fairseq MMS README: "The MMS code and model weights are released under the CC-BY-NC 4.0 license" |
| fairseq repository | MIT | GitHub (the MMS subfolder states CC BY-NC 4.0) |
| uroman | MIT-style permission notice plus a clause that "Any publication of projects using uroman shall acknowledge its use" with a fixed sentence | `LICENSE.txt` |
| `ctc-forced-aligner` code | BSD-2-Clause in `LICENSE`, but `pyproject.toml` declares "CC-BY-NC 4.0" | GitHub |
| `MahmoudAshraf/mms-300m-1130-forced-aligner` | CC BY-NC 4.0 | Hugging Face card |

- Epitran 1.35.3 (PyPI, checked 2026-11-08) is MIT, and its dependencies as locked are permissively licensed: panphon (MIT), regex (Apache-2.0 and CNRI-Python), marisa-trie (MIT; the bundled C++ library is BSD-2-Clause or LGPL-2.1-or-later, used here under BSD-2-Clause), requests (Apache-2.0), jamo (Apache-2.0), unicodecsv (BSD), PyYAML (MIT), editdistance (MIT), munkres (Apache-2.0), pandas (BSD-3-Clause), certifi (MPL-2.0), charset-normalizer (MIT), urllib3 (MIT) and idna (BSD-3-Clause). None is GPL or unlicensed. Epitran is the root dependency group `align`, outside the default groups: `uv sync --all-packages --group align`.
- No GNU General Public License component appears in either chain (MFA, Kaldi, torchaudio, uroman). Checked on 2026-11-07 in each repository: MFA is MIT, Kaldi Apache-2.0, kalpy and `kalpy-kaldi` MIT, Pynini and OpenFst Apache-2.0 (see `tools/README.md`). The XPF licence was not checked (not verified).
- Whether labels produced with CC BY-NC weights carry any obligation is a legal question the licence text does not settle; not verified.

## Maintenance status

- MFA: latest release v3.4.3 (2026-10-08); repository active (last push 2026-10-22).
- MFA 3.4 deprecations: the zip-model download commands go in 5.0 ("sometime in 2027"); in 4.0 ("end of 2026") `mfa align` takes the `align_hf` signature and the old one becomes `mfa align_legacy`, removed in 5.0; MFA tokenisers are removed in 4.0.
- MFA's new paper (Interspeech 2026) is the citation for 3.x models.
- torchaudio: 2.8 deprecated the APIs marked "drop" in issue #3902, 2.9 removed them; `forced_align` was slated for removal but 2.10 (2026-01-21) announced it "preserved and will remain in torchaudio".
- `torchaudio.pipelines.MMS_FA` is still exported on `main` and its docs page carries no deprecation notice; issue #3902 said most of `pipelines` would remain.
- The two CTC alignment tutorials still carry the 2.8/2.9 warning that their APIs were "removed in 2.9", which the 2.10 notes contradict.
- Latest torchaudio is 2.11.1 (2026-10-28); 2.11 "works with torch 2.11 and with every future torch release", so it fits the repo's torch 2.14.1 (not tested).
- fairseq (home of the MMS README) is archived on GitHub; the `MMS_FA` weights download from `dl.fbaipublicfiles.com`, whose long-term availability is not verified.
- uroman: last release 1.3.1.1 (2024-06-27), last push 2024-07.
- Alternatives if torchaudio drops it: `ctc-forced-aligner` (active, last push 2026-09) wraps the same MMS-300M alignment weights converted to Hugging Face Transformers, with its own C++ alignment and uroman; PyPI shows 1.0.2 (2025-02) while the repository's `__version__` is 0.3.0.
- Hugging Face `facebook/mms-1b-all` is the MMS ASR model (CC BY-NC 4.0), not the alignment model.

## Prior use for VAD and segment labels

- Lugosch (Zenodo 2619474, CC BY 4.0): phone and word alignments of all 960 h of LibriSpeech with MFA and its pretrained LibriSpeech model.
- CorentinJ `librispeech-alignments`: MFA word alignments of LibriSpeech, 46 utterances unaligned.
- LibriVAD (Stylianou et al. 2025, arXiv 2512.17281) builds its frame labels and silence insertion from the CorentinJ alignments (its reference [39]), not from Zenodo 2619474 as far as the paper says; its repository ships what looks like the same alignments as Zenodo 2619474 (not compared).
- LibriVAD labels VOiCES by running MFA (`english_us_arpa`) on the clean LibriSpeech source and reusing the labels for the re-recorded versions, "as commonly done in the literature".
- Kraljevski, Tan and Bissiri (Interspeech 2015) compared forced-alignment VAD labels with human labellers and concluded they give "accurate and consistent VAD labels" (numbers not extracted here).
- VoxCommunis provides MFA word and phone TextGrids for the validated clips of Common Voice (84 TextGrid archives on Hugging Face), the closest prior work to our Common Voice labels.
- MMS used its CTC alignment to cut audio into verse segments (about 12 s), a segmentation use, not frame labels.
- No paper using `MMS_FA` or CTC alignment to make VAD frame labels was found (search not exhaustive).

## Recommendation

This section is a proposal, not a sourced fact.
[0025](../project/adr/0025-alignment-labelling.md) labels with pretrained MFA models only (`*_mfa`, `*_cv` or VoxCommunis), and [0023](../project/adr/0023-dataset-composition.md) leaves out the 8 languages that have none: Persian, Urdu, Meadow Mari, Arabic, Laz, Kabardian, Adyghe and Puno Quechua.
No language with a model is held out, so Korean, Vietnamese, Kazakh and Hausa are aligned and trained on.
`MMS_FA` is a researched alternative that was not adopted; the parts below that concern it apply only if it is picked up later.

### Which aligner for which language

- MFA with the Hugging Face `*_mfa` models for the 19 languages that have one (all in the pool; which model labels Czech, Bulgarian and Ukrainian is not chosen, open in [0025](../project/adr/0025-alignment-labelling.md)), with `--use_g2p` for out-of-vocabulary words: the best measured word boundaries of any tool in every English comparison, a 10 ms grid that matches ours, and per-utterance failure metrics that the drop rules of [0025](../project/adr/0025-alignment-labelling.md) already use.
- For the 14 `*_cv` languages, prefer the newer VoxCommunis models (Common Voice 16 to 20) over the Common Voice 7.0 models in mfa-models; `uzbek_cv` (0.24 h, 5 speakers), `abkhaz_cv` (2 h) and `georgian_cv` (6 h) are too small to trust.
- Italian and Dutch have no G2P model, so their out-of-vocabulary words become `spn` instead of getting generated pronunciations (owner, 2026-11-06). Other languages use `--use_g2p`.
- The same VoxCommunis route covers the 10 languages without a model in mfa-models listed above (Catalan, Welsh, Bengali, Estonian, Finnish, Kabyle, Kinyarwanda, Luganda, Cantonese, Mongolian).
- VoxCommunis models need G2P for out-of-vocabulary words: generate pronunciations with the same G2P system their lexicon used (Epitran or XPF) or train one with `mfa train_g2p` on the lexicon, so the phone set matches.
- A language with an acoustic model and no lexicon file (Catalan, so far) gets a rule-based lexicon from Epitran (`dataset/lexicon.py`, `build_lexicon`; XPF only for a language Epitran lacks). A lexicon is only useful if its phones are ones the model knows, so `build_lexicon` takes the phone inventory in the model's `meta.json` and leaves out, and counts, every word with a phone outside it. No phone mapping is applied. The VoxCommunis models were trained on lexicons with their own segmentation (`ca_vxc_acoustic16` names a custom lexicon, `ca_lexicon-IPA`), so the inventory is checked per model. For Catalan, 135 of 293 common words written by hand (a hand-made list, not a corpus) have every Epitran `cat-Latn` phone in the inventory of `ca_vxc_acoustic16`, 46%. That is an upper bound: stress marks and vowel reduction were not checked (`català` is kept as `k a t a l a`, although the model has stressed vowels such as `ˈa`). The words lost are mostly those with `e` (Epitran writes `ɛ`, which the model lacks as a bare phone), a script `ɡ`, a tie-bar affricate or a length mark. On a corpus sample (5,000 Common Voice sentences, 11,549 word types) the share is lower: 35% of types and 47% of tokens, with `ɛ` alone missing from 9,059 of the words left out. The pilot ran this route and Catalan did not calibrate ([MFA pilot](#mfa-pilot-2026-11-08)).
- The remaining 8 (Persian, Urdu, Meadow Mari, Arabic, Laz, Kabardian, Adyghe, Puno Quechua) and the FLEURS languages without an MFA model are left out (0023). `MMS_FA` would cover them, with the costs below.
- Re-align Common Voice 27.0 ourselves rather than reuse the VoxCommunis TextGrids, so that every clip has the same model version and logged metrics; use the shipped TextGrids as a cross-check where clips overlap.

### `MMS_FA` as the alternative (not adopted)

- `MMS_FA` everywhere would remove the aligner as a per-language confound in RQ1, but its word onsets are about twice as far off (43 to 50 ms mean against 20 to 25 ms) and its words about 60 ms too short in English.
- For VAD labels, the 200 ms gap rule hides most of that bias inside utterances; it remains at the first onset and last offset of each speech run, which is where a voicebot's latency and end-of-turn behaviour live.
- If the 8 languages are ever added: MFA where it exists, `MMS_FA` only where it must, and measure the difference instead of avoiding it.
- Run both aligners on a few languages that have an MFA model (English, German and one VoxCommunis language) to measure the MMS bias in onset and offset directly; if the bias is stable, a fixed per-aligner correction (for example widening MMS word spans) can be estimated on the Praat set and applied to all `MMS_FA` languages.
- Record the aligner, model and version per clip in the manifest, and report RQ1 results split by aligner.

### Text normalisation

- MFA: the Hugging Face models' tokenisers handle Japanese and Mandarin; check how MFA treats digits and abbreviations on a Common Voice sample before the full run, since the docs say nothing about it.
- `MMS_FA`: insert the star token at the start and end of each transcript and in place of digits, as the MMS paper did, since Common Voice and VoxPopuli clips can have speech or noise outside the text.
- Cantonese needs a word segmenter and a check that uroman's output is usable; if VoxCommunis covers it, MFA avoids the question.

### Validation

- Run the Praat pilot of [0025](../project/adr/0025-alignment-labelling.md) (100 clips, 25 each from English, German, Italian and Catalan, sampled across score quantiles and labelled blind to MFA; frame agreement on the 10 ms grid, onsets and offsets within 20 and 50 ms) (owner, 2026-11-06). The earlier 500-clip check (125 each for English, German, other `*_mfa` languages and other `*_cv` languages) is deferred.
- Report signed onset and offset errors, not only absolute ones: the expected CTC failure is a bias (late onsets, early offsets), which a shift or padding can correct, while MFA errors are expected to be symmetric.
- If `MMS_FA` is used: map its spans to 10 ms by splitting each 20 ms frame into two, and state that its labels have 20 ms resolution.
- If `MMS_FA` is used: define its drop rules before the run: the MMS paper's greedy-path score above −0.2 as a starting point, plus the duration and coverage rules of [0025](../project/adr/0025-alignment-labelling.md) that do not depend on MFA metrics.
- Run the pilot on one MFA language, one VoxCommunis language and one `MMS_FA` language before setting the pass threshold.

### Licence and tooling

- `MMS_FA` weights are CC BY-NC 4.0; [0024](../project/adr/0024-training-data-licence-policy.md) accepts labelling tools under an NC licence (they are not shipped), but `MMS_FA` is not used.
- The thesis must carry uroman's acknowledgement sentence if `MMS_FA` is used.
- MFA is built from source (owner, 2026-11-06): `tools/montreal-forced-aligner` pinned to v3.4.3 and `tools/kaldi` as git submodules, `just build-mfa` compiling Kaldi into `tools/kaldi/dist` and setting up the uv project `tools/mfa` (editable MFA, `kalpy-kaldi`, a patched `pynini`), run with `just mfa`; conda-forge `kaldi` as `KALDI_ROOT` is the remaining fallback ([0025](../project/adr/0025-alignment-labelling.md)). The exact versions and the reasons for the build steps are in `tools/README.md`.

## MFA pilot (2026-11-08)

The pilot of the spec's quality bar: `index --config pilot`, then `align --config pilot --max-clips 200` (Common Voice only, work dir `work/openvad-pilot`), run on the M3 Pro Mac reading the share over SMB (owner, 2026-11-08), not on the workstation.
Each language took 200 clips in rank order, at most 10 per speaker, the calibration batch's selection, so the calibration batch and the one label batch hold the same 200 clips.
All numbers below are from these runs; the thresholds are a proposal for the controller and the owner, not set in `QualityBar`.

### Coverage

The pilot set is every Common Voice locale extracted on the share (`.<locale>-extracted`, 34 of 53 on 2026-11-08) with over 200 validated hours (`clip_durations.tsv` over `validated.tsv`, no decode) whose row is a `*_cv` or VoxCommunis model or `open`, plus English, German, Japanese and Mandarin.

| Locale | Validated h | Row | In the pilot as |
| :-- | --: | :-- | :-- |
| `ca` | 3,465.7 | open | MFA run, `ca_vxc_acoustic16` by override |
| `en` | 2,723.8 | `english_mfa` | MFA run |
| `rw` | 2,034.0 | open | feasibility row |
| `be` | 1,808.7 | open | feasibility row |
| `de` | 1,385.7 | `german_mfa` | MFA run |
| `kab` | 568.0 | open | feasibility row |
| `eu` | 482.6 | open | feasibility row |
| `ug` | 460.5 | open | feasibility row |
| `lg` | 435.6 | open | feasibility row |
| `ja` | 386.7 | `japanese_mfa` | MFA run |
| `it` | 357.8 | `italian_cv` | MFA run |
| `ba` | 257.8 | open | feasibility row |
| `zh-CN` | 254.1 | `mandarin_mfa` | MFA run |
| `ka` | 251.0 | open | feasibility row |
| `ta` | 234.5 | open | feasibility row |
| `ab` | 204.9 | open | feasibility row |
| `yue` and `zh-HK` | 313.1 (204.1 and 109.0) | open | feasibility row |

- Over 200 h with a `*_mfa` row, so not in the pilot: `es` (619.9 h), `fr` (1,095.3 h), `sw` (407.5 h), `ru` (254.4 h).
- The index maps `zh-HK` to Cantonese (`yue`), as it maps `zh-CN` to Mandarin, so the two Cantonese locales count together.
- `ka`, `ta` and `ab` are over 200 h validated but under 200 h in the train pool (193.2, 182.1 and 157.4 h, [datasets](datasets.md)); under a pool reading they would leave the set. Cantonese would not: its train pool is 264.5 h (182.3 h `yue` and 82.2 h `zh-HK`).
- Missing: no pilot language. The 19 locales not extracted yet (`cs`, `et`, `mn`, `sv-SE`, `ky`, `bn`, `tt`, `cv`, `hy-AM`, `ro`, `el`, `fi`, `bg`, `sr`, `gn`, `hsb`, `vi`, `ha`, `ko`) are all under 200 h in the release statistics of Common Voice 27.0 (an estimate, validated clips times the mean duration; the largest is `cs` with 81.6 h).

### Per-language table

Failure rate is `no_alignment` (MFA wrote no TextGrid or a blank analysis row) over the clips sent to MFA; no clip was dropped before MFA for a missing transcript or a failed decode. Drop shares are of the aligned clips. The fixed rules are the wider beam, word and phone length, coverage and the first 20 ms; the fitted rules are the speaking rate and the log-likelihood floor, which with 200 clips are fitted on the same clips they judge, so the floor removes about 5% by construction.

| Language | Model | Clips | Failure rate | Fixed-rule drops | Fitted-rule drops | Kept | Calibration (clips, speakers, aligned, passing fixed rules) |
| :-- | :-- | --: | --: | :-- | :-- | --: | :-- |
| `de` | `german_mfa` 3.0.0 | 200 | 1 (0.5%) | 9 (4.5%): beam 1, word 2, phone 5, coverage 1 | 12 (6.0%): rate 2, log-likelihood 10 | 178 | 200, 160, 199, 190 |
| `en` | `english_mfa` 3.1.0 | 200 | 5 (2.5%) | 36 (18.5%): beam 13, word 8, phone 13, coverage 1, onset 1 | 8 (4.1%): log-likelihood 8 | 151 | 200, 189, 195, 159 |
| `it` | `italian_cv` 2.0.0 | 200 | 1 (0.5%) | 133 (66.8%): beam 1, word 58, phone 65, onset 9 | 6 (3.0%): rate 2, log-likelihood 4 | 60 | 200, 136, 199, 66 |
| `ja` | `japanese_mfa` 3.0.0 | 200 | 2 (1.0%) | 12 (6.1%): beam 4, word 2, phone 3, coverage 3 | 11 (5.6%): rate 1, log-likelihood 10 | 175 | 200, 163, 198, 186 |
| `zh` | `mandarin_mfa` 3.0.0 | 200 | 4 (2.0%) | 36 (18.4%): beam 3, word 3, phone 23, coverage 7 | 18 (9.2%): rate 10, log-likelihood 8 | 142 | 200, 173, 196, 160 |
| `ca` | `ca_vxc_acoustic16`, Epitran lexicon | 200 | 0 in calibration | 168 (84%) in calibration: beam 9, word 4, phone 155 | not fitted | 0 | 200, 143, 200, 32: `too_few_to_calibrate` |

- Japanese and Mandarin aligned with MFA's tokenisers (sudachipy and spacy-pkuseg in `tools/mfa`): a median of 8 words per kept Mandarin clip and an out-of-vocabulary share of 0.4%. Without them MFA 3.4.3 stops with an `ImportError`, it does not fall back. `sudachipy` 0.7 breaks the Japanese tokeniser (`tools/README.md`).
- Italian fails on length: in the dropped clips the longest phone other than `spn` has a median of 854 ms, so it is not out-of-vocabulary words (their share is 0) but phones of `italian_cv` stretching over pauses. VoxCommunis has a newer `it_vxc_acoustic17`, which the pilot did not try. Running it needs a dictionary route that is not built: `align --model` always builds an Epitran lexicon and cannot use the lexicon VoxCommunis ships for Italian (`lexicons/italian.dict`); that waits for the owner's model choice.
- Catalan: the lexicon built from the 200 clips held 363 of 975 word types; 612 were left out, most because Epitran gives them a phone outside the model's inventory (`ɛ` above all), 90 only because Epitran passed a hyphen or apostrophe through as a phone. The pilot aligned with that lexicon. With the punctuation dropped (fixed after the pilot, without a new MFA run) the same clips give 401 of 975 types (574 left out) and 47.9% of the 1,960 tokens instead of 45.4%. With the 363-word lexicon the left-out words went to `spn`, 155 clips had a phone over 500 ms, and 32 clips passed the fixed rules, under the 50 the calibration needs; the 38 more words would not lift that over 50, but it is not measured.

Throughput, M3 Pro over SMB (MFA's default `--num_jobs 3`, one process decoding):

| Language | Audio per run | MFA s, calibration and batch | Whole round s (with decode) |
| :-- | --: | :-- | :-- |
| `de` | 1,074 s | 54, 53 | 135, 81 |
| `en` | 992 s | 51, 52 | 138, 106 |
| `it` | 1,065 s | 94, 115 | 148, 245 |
| `ja` | 948 s | 107, 102 | 135, 144 |
| `zh` | 970 s | 54, 49 | 94, 81 |
| `ca` | 1,068 s | 25 (calibration only) | 82 |

- A run of 200 clips is dominated by MFA's start. German runs of 20 clips (112.5 s of audio) and 600 clips (3,150 s) took 41.0 and 62.8 s of MFA, so a run costs about 40 s plus about 26 s per hour of audio.
- Decoding the MP3s from the share took 145 s for the 600 clips (0.24 s per clip, about 166 s per audio hour) and 0.4 s per clip on a first read; it, not MFA, bounds a run on this machine.
- Per 1,000 audio hours at German speed, with `align`'s 5 h batches (200 runs): about 2.2 h of fixed cost plus 7.2 h of alignment, so about 9.4 h of MFA, and about 46 h of decoding over SMB in one process. For the 17,329 h train pool of the 27 locales measured in [datasets](datasets.md) (dev and test, the calibration runs and the refill rounds come on top) that is about 7 days of MFA and about 33 days of decoding. Italian and Japanese took about twice as long per 200-clip run; how that splits into fixed and per-hour cost was not measured. Neither is a workstation figure; the throughput gate needs the workstation measurement.
- The first Japanese calibration run took 22 min, mostly before alignment started, while another job read Common Voice transcripts; the Mac froze and restarted during it. The rerun took 107 s.

### Feasibility of the other open languages

No MFA run, no model chosen and no row changed; the choice is the owner's. Every one of them has a VoxCommunis acoustic model and lexicon (`pacscilab/VoxCommunis` on Hugging Face, revision `54d61df1ca273798563e1a48ef9d781ad2a04bc9`); none has an MFA G2P model.

How it was measured, so it can be repeated: the sample is 5,000 sentences of the locale's `validated.tsv`, every k-th row with k = rows // 5,000, through `prepare_transcript` and lower-cased (8,386 to 18,817 word types per language). G2P coverage is the share of those word types (and tokens) that `lexicon.build_lexicon(words, language, model_phones(<VoxCommunis acoustic zip>), transliterate=<g2p>)` keeps, after it drops punctuation (the fix after the pilot): `<g2p>` is `epitran.Epitran(<code>).trans_list` (Epitran 1.35.3) or `xpfcorpus.Transcriber(<code>).transcribe` (xpfcorpus 0.1.1, MIT, the Python interface of the XPF Corpus; run outside the project). Lexicon coverage is the share of the words found, lower-cased, in the VoxCommunis lexicon. The scripts were not committed: they would need a test against Common Voice text.

| Language | VoxCommunis model | `*_cv` in mfa-models | Epitran or XPF code | G2P coverage, types (tokens) | VoxCommunis lexicon coverage, types (tokens) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| `rw` | `rw_epi_acoustic17` | none | Epitran `kin-Latn` | 99.8% (99.9%) | 93% (97%) |
| `be` | `be_xpf_acoustic17` | `belarusian_cv` | XPF `be` | 93% (96%): the model lacks a separate length mark `ː` | 99% (100%) |
| `kab` | `kab_vxc_acoustic20` | none | Epitran `kab-Latn` | 7% (12%): the model lacks a bare `a`, `i`, `u` | 67% (84%) |
| `eu` | `eu_xpf_acoustic17` | `basque_cv` | XPF `eu` | 100% (100%) | 93% (98%) |
| `ug` | `ug_xpf_acoustic17` | `uyghur_cv` | Epitran `uig-Arab` | 19% (27%): the model lacks `a`, `ɛ` | 78% (91%) |
| `lg` | `lg_vxc_acoustic20` | none | Epitran `lug-Latn` | 38% (49%): `ɡ`, `ɾ`, prenasals | 95% (98%) |
| `ba` | `ba_xpf_acoustic16` | `bashkir_cv` | XPF `ba` | 80% (85%): the model lacks `ð`, `θ` | 78% (84%) |
| `ka` | `ka_xpf_acoustic16` | `georgian_cv` | Epitran `kat-Geor` | 34% (39%): `r`, `χ`, affricates | 93% (96%) |
| `ta` | `ta_epi_acoustic17` | `tamil_cv` | Epitran `tam-Taml` | 99.5% (99.7%) | 99% (100%) |
| `ab` | `ab_xpf_acoustic16` | `abkhaz_cv` | XPF `ab` | 48% (56%): `p`, labialised consonants | 28% (43%) |
| `yue` | `yue_epi_acoustic19` (and a toned `yue_vxc_acoustic21_toned`) | none | none for Han script | no word-level G2P for Han script; coverage not measurable | not measured: the text has no spaces and needs a segmenter |

- Where the codes were found: Epitran's codes are the map files of the installed package (`epitran/data/map/<code>.csv`), which has none for Belarusian, Basque, Bashkir or Abkhaz. XPF's are the rule files of the XPF Corpus (`CohenPr-XPF/XPF`, commit `bcc13228ab6a114a9917df25d2b844bd920ce28b`: `Data/be_Belarusan/be.rules`, `Data/eu_Basque/eu.rules`, `Data/ba_Bashkir/ba.rules`, `Data/ab_Abkhaz/ab.rules`), the same codes `xpfcorpus` 0.1.1 takes. XPF's Uyghur is Cyrillic only, while Common Voice writes Uyghur in Arabic script, so `ug` stays on Epitran.
- Han script: Epitran 1.35.3 has `yue-Latn` and `cmn-Latn`, which take romanisations (Jyutping, pinyin), not characters; `xpfcorpus` 0.1.1 has no Cantonese or Mandarin (its only Chinese language is `nan`). So no G2P route for Cantonese characters was found.
- The middle part of a VoxCommunis file name names the G2P system of its lexicon (dataset card): `epi` Epitran, `xpf` the XPF Corpus, `vxc` a custom dictionary.
- Where a lexicon exists, a G2P route is only a fallback for its out-of-vocabulary words. Epitran's segmentation matches the `epi` models (Kinyarwanda, Tamil) and not the `xpf` and `vxc` ones; XPF's matches the Basque model fully and the Belarusian and Bashkir ones mostly.
- The Abkhaz lexicon covers few of the corpus words; whether that is spelling (apostrophes, case) or vocabulary is not checked.

### Threshold proposal

The denominators: the failure rate is over the clips sent to MFA, the drop share over the clips MFA aligned; `QualityBar` should be fed the same.

- Failure rate: at most 5% of the clips sent to MFA. The five languages that calibrated failed 0.5 to 2.5%; a model or text that does not fit shows up in the drop rules rather than here (Catalan failed 0%).
- Drop share: at most 25% of the aligned clips removed by the fixed rules, or, as one number over all drop rules, at most 35%. The aligned languages split into 4.5 to 18.5% (fixed) and 66.8% and above; with 200 clips the standard error at 18.5% is 2.8 points, so 25% is more than two of them above English and Mandarin. Any bar between about 25 and 60% gives the same verdicts here; the low end is proposed because a language that loses a quarter of its clips to fixed rules spends MFA time on labels it throws away. The fitted rules are left out of the preferred form because the log-likelihood floor removes about 5% by construction.
- Verdicts under this proposal: German, English, Japanese and Mandarin pass. Italian fails; its pool is 304.5 h, so the failure rule drops it unless the owner picks another model. Catalan did not calibrate; its pool is 3,409.5 h, over 500 h, so the owner decides.

### Hand check

- `handcheck-sample --config pilot --languages en,de,it` wrote 75 clips (25 each) to `/Volumes/datasets/OpenVAD/datasets/processed/work/openvad-pilot/handcheck/clips` on the share; `key.json` is one level up and `report.json` will be.
- Catalan has no scored clip (`too_few_to_calibrate`), so it is not in the sample and no other language replaces it.

### Machine and versions

Apple M3 Pro (12 cores, 36 GB), macOS 27.2, share over SMB; MFA 3.4.3 (`tools/montreal-forced-aligner` at `93411fc`), Kaldi `f400766`, MFA's SQLite database (`use_postgres: false`), `--num_jobs 3`; sudachipy 0.6.11, sudachidict-core 20260723, spacy-pkuseg 1.0.2, Epitran 1.35.3 (root `align` group), ffmpeg 8.1.3; OpenVAD at `84d0eff` for the runs.
Models with the versions read from them: `english_mfa` 3.1.0 (G2P `english_us_mfa` reads 2.2.6.dev1), `german_mfa` 3.0.0, `italian_cv` 2.0.0 (reads 2.0.0b9), `japanese_mfa` 3.0.0 (reads 3.0; G2P `japanese_katakana_mfa` reads 3.0.0a2.dev0), `mandarin_mfa` 3.0.0, `ca_vxc_acoustic16` at the revision above (SHA-256 `f9524605…e0e3a1`, reads 2.2.17).

## Not verified

- Which MMS variant Rousso et al. 2024 evaluated (ASR output or `MMS_FA`), and that their MFA model is `english_us_arpa`.
- That the `MMS*` row of the MFA 2026 paper comes from `word_align_w2v2.py`.
- The ARPA 3.0 word boundary error behind the mfa-models benchmark text.
- The MFA models used by Weber et al. 2026 for German and Dutch, and the cause of MFA's Dutch result; that paper's acceptance at Interspeech 2026.
- Which model labels Croatian, Bosnian and Serbian; whether `acoustic/arabic.zip` is usable.
- Whether MFA 3.4 runs the other v2.0.0 `*_cv` models and VoxCommunis models than `italian_cv` and `ca_vxc_acoustic16` unchanged, and which MFA version trained the VoxCommunis models other than `ca_vxc_acoustic16` (MFA 2.2.17, from its `meta.json`).
- Versions of the `*_mfa` Hugging Face models other than Polish, Swahili, German and English.
- MFA handling of digits and abbreviations, and what MFA's Japanese and Mandarin tokenisers make of a literal `<unk>`: none of the 1,200 pilot clips has a digit in its transcript.
- uroman's romanisation of Cantonese; whether Adyghe, Kabardian and Puno Quechua were in `MMS_FA` training data; which FLEURS splits it saw.
- `MMS_FA` CPU throughput.
- The XPF licence; the long-term hosting of the `MMS_FA` weights.
- Whether labels made with CC BY-NC weights carry obligations.
- The agreement numbers in Kraljevski et al. 2015.

## Sources

- MFA repository, releases and docs sources: <https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner> (`docs/source/user_guide/configuration/global.rst`, `dictionary.rst`, `troubleshooting.rst`, `workflows/alignment.rst`, `models/model_versions.rst`, `implementations/alignment_analysis.md`, `implementations/fine_tune.md`, `concepts/hmm.md`, `changelog/changelog_3.4.rst`, `installation.rst`, `montreal_forced_aligner/command_line/utils.py`, `config.py`); rendered at <https://montreal-forced-aligner.readthedocs.io>.
- mfa-models repository, model cards and benchmarks: <https://github.com/MontrealCorpusTools/mfa-models> (`docs/source/benchmarks/english_alignments.md`, `docs/source/_static/benchmarks/word_alignment.csv`); <https://mfa-models.readthedocs.io>.
- MFA models on Hugging Face: <https://huggingface.co/MontrealCorpusTools>.
- McAuliffe et al. 2017, "Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi", Interspeech 2017: <https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.pdf>.
- McAuliffe et al. 2026, "Montreal Forced Aligner and the state of speech-to-text alignment in 2026", Interspeech 2026: <https://www.isca-archive.org/interspeech_2026/mcauliffe26_interspeech.pdf>; scripts: <https://github.com/MontrealCorpusTools/mfa-interspeech2026>.
- McAuliffe 2021, "Update on Montreal Forced Aligner performance": <https://memcauliffe.com/update-on-montreal-forced-aligner-performance.html>.
- Pratap et al. 2023, "Scaling Speech Technology to 1,000+ Languages": <https://arxiv.org/abs/2305.13516>.
- fairseq MMS README and data preparation: <https://github.com/facebookresearch/fairseq/tree/main/examples/mms>; language coverage: <https://dl.fbaipublicfiles.com/mms/misc/language_coverage_mms.html>.
- torchaudio source: <https://github.com/pytorch/audio> (`src/torchaudio/pipelines/_wav2vec2/impl.py`, `aligner.py`, `utils.py`, `src/torchaudio/functional/_alignment.py`, `examples/tutorials/ctc_forced_alignment_api_tutorial.py`, `forced_alignment_for_multilingual_data_tutorial.py`, `README.md`).
- torchaudio docs: <https://docs.pytorch.org/audio/main/generated/torchaudio.pipelines.MMS_FA.html>, <https://docs.pytorch.org/audio/main/generated/torchaudio.functional.forced_align.html>.
- torchaudio release notes 2.8.0 to 2.11.1: <https://github.com/pytorch/audio/releases>; maintenance plan: <https://github.com/pytorch/audio/issues/3902>.
- Hwang et al. 2023, "TorchAudio 2.1": <https://arxiv.org/abs/2310.17864> (describes the CUDA `forced_align`, no boundary evaluation).
- Rousso et al. 2024, "Tradition or Innovation: A Comparison of Modern ASR Methods for Forced Alignment", Interspeech 2024: <https://arxiv.org/abs/2406.19363>.
- Huang et al. 2024, "Less peaky and more accurate CTC forced alignment by label priors": <https://arxiv.org/abs/2406.02560>.
- Weber et al. 2026, "Multilingual Word-Level Forced Alignment with Self-Supervised Representations and Learned Dynamic Programming": <https://arxiv.org/abs/2606.10675>.
- uroman: <https://github.com/isi-nlp/uroman>, <https://pypi.org/project/uroman/>.
- `ctc-forced-aligner`: <https://github.com/MahmoudAshraf97/ctc-forced-aligner>; weights: <https://huggingface.co/MahmoudAshraf/mms-300m-1130-forced-aligner>.
- `facebook/mms-1b-all`: <https://huggingface.co/facebook/mms-1b-all>.
- VoxCommunis: <https://huggingface.co/datasets/pacscilab/VoxCommunis>, <https://osf.io/t957v/>; Ahn and Chodroff 2022, LREC: <http://www.lrec-conf.org/proceedings/lrec2022/pdf/2022.lrec-1.566.pdf>.
- Epitran: <https://github.com/dmort27/epitran>; XPF: <https://github.com/CohenPr-XPF/XPF>.
- LibriSpeech alignments (Lugosch): <https://zenodo.org/records/2619474>; CorentinJ: <https://github.com/CorentinJ/librispeech-alignments>.
- Stylianou et al. 2025, "LibriVAD": <https://arxiv.org/abs/2512.17281>.
- Kraljevski et al. 2015, "Comparison of forced-alignment speech recognition and humans for generating reference VAD": <https://www.isca-archive.org/interspeech_2015/kraljevski15_interspeech.html>.
