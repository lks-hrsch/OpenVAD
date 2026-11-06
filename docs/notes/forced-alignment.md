# Forced alignment

A comparison of the Montreal Forced Aligner (MFA 3.x with its pretrained models) and the MMS forced aligner in torchaudio (`MMS_FA`) as labelling tools for OpenVAD.
State of 2026-11-04.

The labels we need: word-level speech intervals from forced alignment, turned into 10 ms speech/non-speech frames, where a gap between two words shorter than the proposed 200 ms counts as speech and a longer one as non-speech ([0022](../project/adr/0022-frame-labels.md)).
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
- Whether MFA 3.4 runs these v2.0.0 models unchanged is not verified.

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
- MFA installs from conda-forge (Kaldi, OpenFst, Pynini); a `Montreal-Forced-Aligner` 3.4.3 and a `kalpy-kaldi` package exist on PyPI, but a pip-only install is not verified.
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

- No GNU General Public License component appears in either chain (MFA, Kaldi, torchaudio, uroman); Kaldi's and Pynini's licences were not checked here (not verified).
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
- The same VoxCommunis route covers the 10 languages without a model in mfa-models listed above (Catalan, Welsh, Bengali, Estonian, Finnish, Kabyle, Kinyarwanda, Luganda, Cantonese, Mongolian).
- VoxCommunis models need G2P for out-of-vocabulary words: generate pronunciations with the same G2P system their lexicon used (Epitran or XPF) or train one with `mfa train_g2p` on the lexicon, so the phone set matches.
- The remaining 8 (Persian, Urdu, Meadow Mari, Arabic, Laz, Kabardian, Adyghe, Puno Quechua) and the FLEURS languages without an MFA model are left out (0023). `MMS_FA` would cover them, with the costs below.
- Re-align Common Voice 27.0 ourselves rather than reuse the VoxCommunis TextGrids, so that every clip has the same model version and logged metrics; use the shipped TextGrids as a cross-check where clips overlap.

### `MMS_FA` as the alternative (not adopted)

- `MMS_FA` everywhere would remove the aligner as a per-language confound in RQ1, but its word onsets are about twice as far off (43 to 50 ms mean against 20 to 25 ms) and its words about 60 ms too short in English.
- For VAD labels, the proposed 200 ms gap rule hides most of that bias inside utterances; it remains at the first onset and last offset of each speech run, which is where a voicebot's latency and end-of-turn behaviour live.
- If the 8 languages are ever added: MFA where it exists, `MMS_FA` only where it must, and measure the difference instead of avoiding it.
- Run both aligners on a few languages that have an MFA model (English, German and one VoxCommunis language) to measure the MMS bias in onset and offset directly; if the bias is stable, a fixed per-aligner correction (for example widening MMS word spans) can be estimated on the Praat set and applied to all `MMS_FA` languages.
- Record the aligner, model and version per clip in the manifest, and report RQ1 results split by aligner.

### Text normalisation

- MFA: the Hugging Face models' tokenisers handle Japanese and Mandarin; check how MFA treats digits and abbreviations on a Common Voice sample before the full run, since the docs say nothing about it.
- `MMS_FA`: insert the star token at the start and end of each transcript and in place of digits, as the MMS paper did, since Common Voice and VoxPopuli clips can have speech or noise outside the text.
- Cantonese needs a word segmenter and a check that uroman's output is usable; if VoxCommunis covers it, MFA avoids the question.

### Validation

- Keep the Praat check of [0025](../project/adr/0025-alignment-labelling.md) (500 clips: 125 each for English, German, other `*_mfa` languages and other `*_cv` languages; frame agreement on the 10 ms grid, onsets and offsets within 20 and 50 ms), reported per group, with no per-language quota.
- Report signed onset and offset errors, not only absolute ones: the expected CTC failure is a bias (late onsets, early offsets), which a shift or padding can correct, while MFA errors are expected to be symmetric.
- If `MMS_FA` is used: map its spans to 10 ms by splitting each 20 ms frame into two, and state that its labels have 20 ms resolution.
- If `MMS_FA` is used: define its drop rules before the run: the MMS paper's greedy-path score above −0.2 as a starting point, plus the duration and coverage rules of [0025](../project/adr/0025-alignment-labelling.md) that do not depend on MFA metrics.
- Run the pilot on one MFA language, one VoxCommunis language and one `MMS_FA` language before setting the pass threshold.

### Licence and tooling

- `MMS_FA` weights are CC BY-NC 4.0; [0024](../project/adr/0024-training-data-licence-policy.md) accepts labelling tools under an NC licence (they are not shipped), but `MMS_FA` is not used.
- The thesis must carry uroman's acknowledgement sentence if `MMS_FA` is used.
- MFA comes from conda-forge, outside the repo's uv workflow; keep it as a separate labelling environment and record its exact versions, or test whether the PyPI packages suffice.

## Not verified

- Which MMS variant Rousso et al. 2024 evaluated (ASR output or `MMS_FA`), and that their MFA model is `english_us_arpa`.
- That the `MMS*` row of the MFA 2026 paper comes from `word_align_w2v2.py`.
- The ARPA 3.0 word boundary error behind the mfa-models benchmark text.
- The MFA models used by Weber et al. 2026 for German and Dutch, and the cause of MFA's Dutch result; that paper's acceptance at Interspeech 2026.
- Which model labels Croatian, Bosnian and Serbian; whether `acoustic/arabic.zip` is usable.
- Whether MFA 3.4 runs the v2.0.0 `*_cv` models and the newer VoxCommunis models unchanged, and which MFA version trained the latter.
- Versions of the `*_mfa` Hugging Face models other than Polish, Swahili, German and English.
- MFA handling of digits and abbreviations.
- uroman's romanisation of Cantonese; whether Adyghe, Kabardian and Puno Quechua were in `MMS_FA` training data; which FLEURS splits it saw.
- `MMS_FA` CPU throughput; a pip-only MFA install.
- XPF, Kaldi and Pynini licences; the long-term hosting of the `MMS_FA` weights.
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
