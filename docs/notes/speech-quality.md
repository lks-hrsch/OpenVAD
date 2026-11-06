# Speech quality and recording condition

A comparison of automatic tools that estimate the quality or the recording condition of a speech clip, as per-clip metadata for the OpenVAD manifest.
State of 2026-11-04.

The uses: drop broken clips, stratify results, and estimate the unknown baseline SNR of crowd-sourced Common Voice clips before noise is mixed in ([0023](../project/adr/0023-dataset-composition.md)).
The data: Common Voice 27.0 (≈ 5 s read sentences, phone and laptop microphones, 53 locales), MLS, FLEURS, VoxPopuli, AMI, CallHome (8 kHz content upsampled to 16 kHz), SWC, Speech Commands and VOiCES; the size of the pool is in [datasets](datasets.md#pool); there will be a Small and a Large set, each with a train, validation and test split, and Small is a subset of Large ([0023](../project/adr/0023-dataset-composition.md)).
Labelling and filtering tools are not shipped ([0024](../project/adr/0024-training-data-licence-policy.md)), so their licences matter only for running them; ONNX Runtime is allowed in `benchmarking/` only ([0001](../project/adr/0001-no-onnx-in-engine.md)).
Every section before [Recommendation](#recommendation) states sourced facts only; the recommendation is a proposal for the owner.
Anything not checked against a primary source is marked "not verified".

## Summary

| Tool | Output | Trained or calibrated on | Runtime | Code licence | Weights licence | Last change |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| DNSMOS (Deep Noise Suppression Mean Opinion Score) P.835 (`microsoft/DNS-Challenge`) | SIG, BAK, OVRL (1 to 5) | ≈ 75 h of P.835 ratings, noise-suppressed clips, 16 kHz | ONNX Runtime | MIT | not stated (repository: code MIT, content CC BY 4.0) | 2024-03-29 |
| DNSMOS P.808 (same repository) | one MOS | ≈ 300 h of P.808 ratings, mostly English | ONNX Runtime | MIT | as above | 2024-03-29 |
| NISQA (Non-Intrusive Speech Quality Assessment) v2.0 | MOS, noisiness, colouration, discontinuity, loudness | 59 training sets (72,903 files), telephone and VoIP distortions | PyTorch | MIT | CC BY-NC-SA 4.0 | 2024-12-01 |
| Squim objective (torchaudio) | estimated STOI, WB-PESQ, SI-SDR | DNS Challenge 2020 mixtures, SNR -15 to 25 dB, 5 s, 16 kHz | PyTorch | BSD-2-Clause | CC BY 4.0 | torchaudio 2.11.1, 2026-10-28 |
| Squim subjective (torchaudio) | MOS, needs a clean non-matching reference | BVCC and DAPS | PyTorch | BSD-2-Clause | CC BY-NC 4.0 | as above |
| UTMOS22 (UTokyo-SaruLab Mean Opinion Score, 2022 system) | naturalness MOS | VoiceMOS 2022 (synthetic speech, BVCC) | PyTorch, fairseq | MIT | not verified | 2024-04-11 |
| UTMOSv2 | naturalness MOS | BVCC, Blizzard, SOMOS, sarulab-data (synthetic speech) | PyTorch | MIT | in the MIT repository | v1.3.0, 2026-02-23 |
| Audiobox Aesthetics | production quality, production complexity, content enjoyment, content usefulness (1 to 10) | ≈ 500 h, 97k clips of speech, sound and music | PyTorch | CC BY 4.0 (parts MIT) | CC BY 4.0 | 2025-06-05 |
| Brouhaha | frame-wise VAD, SNR and C50 | 1,250 h of LibriSpeech contaminated with AudioSet noise and real RIRs, SNR 0 to 30 dB | PyTorch, pyannote.audio 3.x | MIT | MIT (GitHub checkpoint); OpenRAIL, gated (Hugging Face `pyannote/brouhaha`) | 2026-05-23 |
| WADA-SNR (Waveform Amplitude Distribution Analysis signal-to-noise ratio) | one SNR per clip | an algorithm; gamma speech model, Gaussian noise | NumPy | own code | none | 2008 paper |
| `pyloudnorm` | integrated loudness (loudness units relative to full scale), International Telecommunication Union Radiocommunication Sector BS.1770-4 | a standard | NumPy, SciPy | MIT | none | v0.2.0, 2026-01-04 |
| ASR agreement | CER or WER against the transcript | depends on the ASR model | any | depends | depends | |

## What each tool outputs and what it was trained on

### DNSMOS P.835 and P.808

- DNSMOS P.835 predicts three scores on the ITU-T P.835 scale: speech quality (SIG), background noise quality (BAK) and overall quality (OVRL).
- It was trained on DNS Challenge 3 P.835 ratings of noisy clips processed by about 40 noise suppressors, plus internally processed clips: about 30,000 clips, ≈ 9 s each, ≈ 75 h.
- The training scores are skewed: most SIG and OVRL ratings lie between 3 and 4, and BAK is skewed towards above 4.
- The input is a log power spectrogram over 9 s at 16 kHz (20 ms frames, 10 ms hop); the model is a CNN.
- The authors deliberately avoid feature normalisation so that the model captures the lower ratings humans give to quiet clips.
- DNSMOS P.808 (the first paper) predicts one overall MOS from a 120-band log Mel spectrogram over 9 s at 16 kHz, trained on about 120,000 rated clips (≈ 300 h per the P.835 paper's Table 1).
- Its training data was predominantly English with little or no emotional speech and no singing.
- On the DNS Challenge 2 blind set, the per-category Spearman rank correlation coefficient of the first model was 0.78 for English, 0.67 non-English, 0.51 tonal; later self-teaching stages reached 0.81 to 0.93 English and 0.85 to 0.92 non-English.
- Which training stage the released `model_v8.onnx` is, is not stated (not verified).
- `dnsmos_local.py` writes `SIG`, `BAK`, `OVRL`, their `_raw` versions before a per-model polynomial mapping, and `P808_MOS`; `-p` selects the personalised model (pDNSMOS), which penalises interfering talkers.

### NISQA

- NISQA v2.0 predicts overall quality and the four quality dimensions noisiness, colouration (`Coloration`), discontinuity and loudness, for speech sent through a communication system.
- A separate model, NISQA-TTS, predicts the naturalness of synthetic speech from text-to-speech systems.
- Training: 59 datasets (72,903 files), validation: 18 datasets (9,567 files), test: 4 datasets (952 files); 55 come from the POLQA (Perceptual Objective Listening Quality Assessment) pool, 7 from ITU-T P Suppl. 23.
- The new NISQA training sets use English source speech (AusTalk, LibriVox via DNS Challenge, TSP, UK-Ireland) with simulated packet loss, bandpass filtering, codecs, clipping and DNS Challenge noise, plus live Skype and phone recordings.
- The evaluated datasets span Swedish, French, English, German, Dutch, Chinese and Japanese, in narrowband (NB), super-wideband (SWB) and fullband (FB) conditions (paper Table 4).
- The architecture is CNN, self-attention and attention pooling on 48-band Mel spectrogram segments of 150 ms.

### TorchAudio-Squim

- `SQUIM_OBJECTIVE` estimates STOI, wideband PESQ and SI-SDR without a reference.
- It was trained on DNS Challenge 2020 data: clean speech mixed with noise at -15 to 25 dB SNR, part of it processed by three GCRN enhancement systems; 364,500 training, 14,600 validation and 22,800 test samples, truncated to 5 s.
- The model is a DPRNN encoder with one transformer branch per metric; 7.39 M parameters, 40.27 G multiply-accumulate operations per 5 s clip.
- `SQUIM_SUBJECTIVE` is NORESQA-MOS: it predicts MOS from the test clip plus any clean speech clip as a non-matching reference, trained on BVCC and DAPS.

### UTMOS22 and UTMOSv2

- Both predict the naturalness MOS of synthetic speech, from the VoiceMOS Challenges 2022 and 2024.
- UTMOS22 was trained on BVCC (Blizzard and Voice Conversion Challenge samples, ESPnet-TTS) for the main track and on Chinese synthetic speech for the out-of-domain track; it ensembles fine-tuned self-supervised learning models and simple learners on self-supervised learning features.
- UTMOSv2 fuses an EfficientNetV2 spectrogram branch and a `facebook/wav2vec2-base` self-supervised learning branch, trained on BVCC, Blizzard Challenge 2008 to 2011, SOMOS and sarulab-data (all synthetic-speech listening tests).
- The UTMOSv2 configuration samples 16 kHz audio, takes 3 s for the self-supervised learning branch and two 1.4 s spectrogram frames; the inference guide offers `num_repetitions` for more accurate predictions.

### Meta Audiobox Aesthetics

- Predicts four axes for speech, sound and music: production quality (PQ), production complexity (PC), content enjoyment (CE) and content usefulness (CU).
- Training: about 500 h, 97k clips, evenly split between speech, sound and music, mostly 10 to 30 s, loudness-normalised before annotation, 3 ratings per clip.
- The encoder is WavLM-like (12 transformer layers, 768 hidden); input is resampled to 16 kHz mono; inference uses 10 s sliding windows with a weighted average.
- The evaluation set AES-Natural includes 500 Common Voice 13.0 clips; the authors resampled them to 16 kHz because many are not full-band.
- The authors name data filtering as a target use.

### Brouhaha

- Brouhaha outputs, per 16 ms frame, a speech probability, an SNR and a C50 (early-to-late energy ratio of the room response, a reverberation measure).
- Training: 1,000 h of LibriSpeech (English read speech) with non-speech padded to ≈ 30%, contaminated with ≈ 1,500 h of AudioSet noise (human vocalisations removed) and 385 real impulse responses (EchoThief, MIT Acoustical Reverberation Scene); 1,250 h of synthetic audio, 16 kHz mono.
- Mixing SNRs were drawn between 0 and 30 dB; the frame SNR label comes from a 2 s sliding window.
- The model is SincNet, three bidirectional LSTM layers of 256 and feed-forward layers, trained on 6 s segments; the SNR output is a sigmoid scaled to -15 to 80 dB, C50 to -10 to 60 dB.
- The LSTMs are bidirectional, so the model is not causal; this does not matter for offline labelling.

### WADA-SNR and the National Institute of Standards and Technology STNR

- WADA-SNR estimates one SNR per clip from the waveform amplitude distribution: clean speech is modelled as gamma-distributed with shape 0.4, noise as Gaussian, and a statistic of the noisy waveform maps to SNR through a precomputed table.
- Zero-valued samples break the log in the statistic; the paper drops them or replaces them by a small value.
- STNR (the signal-to-noise ratio estimator of the National Institute of Standards and Technology) estimates a peak SNR from a histogram of short-time energy (95th percentile of signal power against 50th percentile of noise power); per the WADA paper this overestimates the SNR by about 7 dB for Gaussian noise.

### Signal checks

- `pyloudnorm` implements International Telecommunication Union Radiocommunication Sector BS.1770-4 with 400 ms gating blocks and K-weighting; `integrated_loudness` validates the input length against the block size.
- Clipping, DC offset, digital silence and effective bandwidth have no standard tool; they are simple own code (see [Recommendation](#recommendation)).
- NISQA lists bandpass filtering among its simulated distortions, and the Audiobox Aesthetics authors note that many Common Voice clips are not full-band.

### ASR agreement

- VoxPopuli filtered its transcribed segments with a maximum CER of 20%, using its own ASR model on the official transcripts.
- The People's Speech removed aligned samples whose CER against the transcript exceeded 50% (2,050 h removed, 31,400 h kept) and trained its baseline on a 20,000 h subset with at most 20% CER.
- VoxPopuli ships an ASR hypothesis with CER and WER per segment ([datasets](datasets.md#voxpopuli)).
- MOSEL (`FBK-MT/mosel`; from the title "950,000 Hours of Speech Data for Open-Source Speech Foundation Model Training on EU Languages"; its authors give no expansion) ships hallucination flags.

### Other tools found

- DNSMOS Pro (`fcumlin/DNSMOSPro`, MIT, Interspeech 2024): a reduced-size model that outputs a Gaussian MOS distribution, TorchScript weights trained on VCC2018, BVCC or NISQA; 16 kHz, 10 s recommended.
- SCOREQ (`alessandroragano/scoreq`, NeurIPS 2024): MOS for natural speech (trained on NISQA TRAIN SIM) or synthetic speech (VoiceMOS 2022); ONNX Runtime by default, PyTorch as an extra; the README says MIT but the repository has no licence file and GitHub reports none.
- VERSA (`wavlab-speech/versa`): a toolkit wrapping 90+ metrics, including several of the above; GitHub reports Apache-2.0 while the README badge says MIT.
- `torchmetrics` wraps DNSMOS, downloads the same three ONNX files from the DNS-Challenge repository and needs `onnxruntime`.

## Published accuracy

All correlations are against human ratings or synthetic ground truth on the authors' own test sets; none is on Common Voice, multilingual crowd-sourced speech or upsampled telephone audio.

### Quality predictors

| Tool | Test set | Level | Correlation |
| :-- | :-- | :-- | :-- |
| DNSMOS P.835 | 850 unseen clips, 17 suppressors, incl. non-English and tonal | per model | Pearson correlation coefficient SIG 0.94, BAK 0.98, OVRL 0.98; Spearman rank correlation coefficient 0.95, 0.99, 0.98 |
| DNSMOS P.835 | same | per clip | Pearson correlation coefficient SIG 0.71, BAK 0.83, OVRL 0.82; Spearman rank correlation coefficient 0.72, 0.82, 0.81 |
| DNSMOS P.808 | DNS Challenge test set | per model | Pearson correlation coefficient 0.93, Spearman rank correlation coefficient 0.94 (PESQ 0.78 / 0.82) |
| NISQA v2.0 | NISQA TEST P501 / FOR / NSC / LIVETALK | per condition | MOS Pearson correlation coefficient 0.95 / 0.95 / 0.97 / 0.90 |
| NISQA v2.0 | NB datasets (POLQA pool, P Suppl. 23) | per condition | MOS Pearson correlation coefficient 0.77 to 0.98 |
| NISQA v2.0 | NISQA TEST LIVETALK (real phone calls, German) | per condition | noisiness 0.76, colouration 0.87, discontinuity 0.83, loudness 0.71 |
| Squim objective | DNS 2020 synthetic test | per clip | STOI MAE 1.994%, Pearson correlation coefficient 0.950; PESQ MAE 0.142, Pearson correlation coefficient 0.958; SI-SDR MAE 0.838 dB, Pearson correlation coefficient 0.985 |
| UTMOS22 strong learner | VoiceMOS 2022 main (BVCC) | utterance / system | Spearman rank correlation coefficient 0.881 / 0.925 |
| UTMOSv2 | VoiceMOS 2024 track 1 (25% zoom-in) | utterance / system | Spearman rank correlation coefficient 0.613 / 0.919 |
| Audiobox Aesthetics PQ | VMC22 main / out-of-domain (Chinese) | utterance Pearson correlation coefficient | 0.689 / 0.651 |
| Audiobox Aesthetics PQ | AES-Natural speech (EARS, LibriTTS, Common Voice 13.0) | utterance Pearson correlation coefficient with human PQ | 0.888 (DNSMOS 0.662, Squim 0.660, UTMOSv2 0.603) |

- The P.835 authors note that two P.835 runs on the same clips agree only at Pearson correlation coefficient 0.7 to 0.8 because each clip has only 5 ratings, which caps per-clip accuracy.
- For P.808, the same figure is about 0.5.
- In the Audiobox Aesthetics comparison, DNSMOS means the P.808 MOS and Squim means its PESQ estimate; the authors chose P.808 over P.835 because P.835 is tailored to speech enhancement.
- Audiobox Aesthetics flags data leakage for UTMOSv2 on VMC22 main (UTMOSv2 also trained on the BVCC test set).
- Audiobox Aesthetics training data is mostly English; the authors read the Chinese out-of-domain result as robustness.

### SNR and reverberation estimators

| Tool | Test set | Result |
| :-- | :-- | :-- |
| Brouhaha | own synthetic test (LibriSpeech test, unseen noise and RIRs), speech frames | SNR MAE 4.1 dB; C50 MAE 3.5 dB (Table 1) |
| Brouhaha | same, predicted and true SNR clipped to [-15, 30] dB | SNR MAE 2.3 dB; oracle-VAD heuristic 8.4 dB; Li et al. 12.5 dB |
| Brouhaha | BUT Speech@FIT Reverb, 1,000 utterances | C50 R² 0.85, MAE 1.1 dB |
| Brouhaha | VAD F-score, synthetic / BabyTrain (real child-centred) | 93.7% / 77.2% (pyannote.audio 89.0% / 80.8%) |
| WADA-SNR | 1,600 DARPA RM utterances with white noise, music, one interfering talker | less bias and less dependence on noise type than National Institute of Standards and Technology STNR; much smaller spread below 20 dB, larger spread above |
| Squim SI-SDR | DNS 2020 synthetic test | MAE 0.838 dB, Pearson correlation coefficient 0.985 |

- The WADA-SNR paper gives no error figure in its text, only plots; the text says noise was added at 10 to 30 dB, while the plotted axes run from -10 to 30 dB.
- Brouhaha has no SNR evaluation on real recordings, since real SNRs are unknown; the paper shows on real data only that Whisper's word hit rate falls with predicted SNR decile (83% at 12 to 24 dB, 38% at -9 to -4 dB).

## Input requirements and narrowband audio

| Tool | Sample rate | Clip length | Short clips (≈ 5 s Common Voice) |
| :-- | :-- | :-- | :-- |
| DNSMOS | resamples to 16 kHz (`librosa`) | 9.01 s windows, 1 s hop, mean over windows | the script repeats a clip end to end until it reaches 9.01 s |
| NISQA | resamples to the model rate (dataset class defaults: 48 kHz, Mel up to 16 kHz; the paper says 20 kHz) | up to 52 s with `ms_max_segments: 1300` | error below 15 Mel frames (about 150 ms) |
| Squim objective | 16 kHz, caller resamples | trained on 5 s; Data-Speech crops to the first 15 s | fits the training length |
| UTMOSv2 | 16 kHz default | 3 s self-supervised learning crop, two 1.4 s spectrogram frames | fits |
| Audiobox Aesthetics | resamples to 16 kHz mono | 10 s sliding windows | fits |
| Brouhaha | 16 kHz | trained on 6 s chunks, sliding windows | fits |
| WADA-SNR | any | needs enough samples for the averages | not studied |
| `pyloudnorm` | any | at least one 400 ms block | fits |

- No tool documents its behaviour on 8 kHz content upsampled to 16 kHz (not verified for all).
- NISQA is the only one trained on narrowband conditions by design, so it scores a bandwidth limit as a quality loss.
- DNSMOS and Squim were trained on 16 kHz DNS Challenge data, Brouhaha on 16 kHz LibriSpeech; how they score CallHome or narrowband Common Voice phones is not verified.
- DNSMOS uses no feature normalisation, so its scores depend on the playback level.
- `dnsmos_local.py` calls `librosa.resample(aud, input_fs, fs)` with positional rates, which current `librosa` releases reject (not verified on a current install).

## Runtime and dependencies

- DNSMOS: ONNX Runtime, `librosa`, `soundfile`; model files 1.16 MB (`sig_bak_ovr.onnx`) and 0.22 MB (`model_v8.onnx`); the script runs clips in a thread pool.
- In this repository ONNX Runtime is allowed only in `benchmarking/`, so DNSMOS needs the owner's exception or a conversion to PyTorch.
- NISQA: PyTorch; the pinned conda environment is old (Python 3.9.9, PyTorch 1.10.1, `librosa` 0.8.1); weights about 1 MB; batch size and DataLoader workers are configurable.
- Squim: part of torchaudio; weights 28.2 MB (objective) and 360 MB (subjective).
- torchaudio is in a maintenance phase since 2.8, with decoding moved to TorchCodec; `models.squim` and the Squim pipelines are still present at v2.11.1.
- UTMOSv2: PyTorch, `timm`, `transformers`, `librosa`, `torchaudio`; weights fetched by a sparse `git` checkout of the repository or from Hugging Face.
- UTMOS22: fairseq checkpoints; the quick path clones a Hugging Face Space.
- Audiobox Aesthetics: `pip install audiobox_aesthetics`, Python 3.9+, PyTorch 2.2+; a CLI with batch size and SLURM array options.
- Brouhaha: `pyannote.audio>=3.1,<3.3.1`, `speechbrain>=0.5.14,<1.0`, `numpy<2.0`; its CI tests installation on Python 3.8 only; checkpoint 47.2 MB in the repository.
- OpenVAD's workspace pins Python 3.12 and `torch==2.14.1`; whether Brouhaha's pins resolve there is not verified.
- Documented throughput: only Squim's MAC count and Emilia-Pipe's whole-pipeline rate (about 2.5 h of raw audio per minute on eight RTX 4090, including source separation, diarisation, ASR and DNSMOS).
- GPU and CPU throughput of each tool alone is not documented (not verified).

## Licences

| Tool | Code | Weights | Note |
| :-- | :-- | :-- | :-- |
| DNSMOS | MIT (`LICENSE-CODE`) | not stated; documentation and content are CC BY 4.0 (`LICENSE`) | the README grants MIT for code and CC BY 4.0 for other content |
| NISQA | MIT | CC BY-NC-SA 4.0 (`weights/LICENSE_model_weights`) | non-commercial and share-alike |
| Squim objective | BSD-2-Clause (torchaudio) | CC BY 4.0, pointing to the DNS Challenge licence | |
| Squim subjective | BSD-2-Clause | CC BY-NC 4.0 | |
| UTMOS22 | MIT | not verified | weights in a Hugging Face Space |
| UTMOSv2 | MIT | weights in the MIT repository's `models` directory | training data includes SOMOS (CC BY-NC-SA 4.0) |
| Audiobox Aesthetics | CC BY 4.0; WavLM-derived parts MIT | CC BY 4.0 (Hugging Face card) | |
| Brouhaha | MIT (CNRS, 2020) | GitHub checkpoint: in the MIT repository; Hugging Face `pyannote/brouhaha`: OpenRAIL, gated (automatic approval) | dependencies MIT (`pyannote.audio`), Apache-2.0 (`speechbrain`) |
| `pyloudnorm` | MIT | none | |
| DNSMOS Pro | MIT | in the repository | |
| SCOREQ | README says MIT, no licence file | Zenodo, not verified | |

- No GNU General Public License was found for any of these tools or their direct dependencies listed here.
- Whether metadata produced with NC or SA weights (NISQA, Squim subjective) carries obligations is not verified; [0024](../project/adr/0024-training-data-licence-policy.md) accepts labelling tools under an NC licence because they are not shipped, and says nothing about such metadata.

## Maintenance status

| Tool | Latest | Archived |
| :-- | :-- | :-- |
| DNS-Challenge (DNSMOS) | last DNSMOS change 2024-03-29 (removed the deprecated web API script); last push 2024-07-25; no releases | no |
| NISQA | last commit 2024-12-01 (error for too short files); no releases | no |
| torchaudio (Squim) | v2.11.1, 2026-10-28 | no |
| UTMOS22 | last commit 2024-04-11 | no |
| UTMOSv2 | v1.3.0, 2026-02-23; last commit 2026-04-02 | no |
| Audiobox Aesthetics | PyPI 0.0.4, 2025-05-01; last commit 2025-06-05 | no |
| Brouhaha | last commit 2026-05-23 (NumPy pin); Hugging Face model last modified 2022-11-15 | no |
| `pyloudnorm` | v0.2.0, 2026-01-04 | no |
| Data-Speech | last push 2024-09-03 | no |
| SCOREQ | v1.0.0, 2025-06-25 (PyPI 1.0.1) | no |
| VERSA | v1.1.0, 2026-09-09 | no |

## Known pitfalls

- DNSMOS was built to rank noise suppressors; its per-model correlations (0.94 to 0.98) are much higher than its per-clip ones (0.71 to 0.83), and a manifest needs per-clip scores.
- DNSMOS training BAK is skewed above 4, and SIG and OVRL cluster between 3 and 4, so clean read speech may saturate (inferred from the training distribution, not verified on Common Voice).
- DNSMOS repeats short clips to 9.01 s, so a 5 s Common Voice clip is scored on a looped signal.
- DNSMOS depends on level by design, so a quiet but clean phone recording scores lower.
- DNSMOS's first model generalised poorly to non-English and tonal speech (Spearman rank correlation coefficient 0.67 and 0.51 against 0.78 English); the later stages narrowed the gap.
- NISQA targets transmitted speech; its source speech for the new training sets is English; its weights are non-commercial share-alike.
- Squim objective was trained only on DNS 2020 synthetic mixtures and enhanced speech; its accuracy on real recordings is not published, and its SI-SDR is a distortion measure, not an SNR.
- UTMOS22 and UTMOSv2 predict naturalness of synthetic speech from text-to-speech and voice conversion listening tests; on natural speech UTMOSv2 correlated less with human production quality than DNSMOS and Squim in the Audiobox comparison (0.603 against 0.662 and 0.660).
- Audiobox Aesthetics scores production and content, so music, emotion and content style move the score as well as noise.
- Brouhaha's SNR labels lie between 0 and 30 dB; the SNR estimates Data-Speech bins for MLS and LibriTTS-R run from 17.1 to 75.0 dB, and the Parler-TTS paper plots MLS estimates mostly between 30 and 70 dB, so values above 30 dB are outside the training range (my reading).
- Brouhaha's real-data VAD F-score (77.2%) is below pyannote.audio's (80.8%); its frame VAD is a cross-check, not a label source.
- WADA-SNR assumes Gaussian noise and one stationary speaker; its spread grows above 20 dB, where National Institute of Standards and Technology STNR is tighter.
- All MOS predictors here were trained on mostly English source speech; published non-English numbers exist only for DNSMOS (DNS Challenge 2 categories), NISQA (POLQA pool and German test sets) and Audiobox Aesthetics and UTMOS22 (Chinese out-of-domain).
- No primary source found measures accent bias in these predictors (not verified).

## Prior use as data filters

| Corpus | Tool | Rule |
| :-- | :-- | :-- |
| Emilia (Emilia-Pipe) | DNSMOS P.835 OVRL | keep segments with OVRL above 3.0; also Whisper language ID with confidence of at least 80%, and average character duration outliers beyond 1.5 IQR dropped |
| LibriTTS | WADA-SNR | drop utterances below 20 dB from the clean subsets and below 0 dB from the other subsets; about 25% of clean sentences fell to the SNR rule (65,718) |
| Libri-Light | VAD-based SNR | per-file SNR from VAD posteriors stored as metadata for selection, not a filter |
| Parler-TTS (Data-Speech) | Brouhaha SNR and C50; Squim PESQ, SI-SDR, STOI | continuous values binned into seven text bins for descriptions; "very good recording" and "very bad recording" when SNR and C50 are both in the top or bottom bin; the README suggests filtering on them but sets no threshold |
| VoxPopuli | ASR CER | transcribed segments kept at CER up to 20% |
| The People's Speech | ASR CER | alignments with CER above 50% removed; baseline trained on a 20,000 h subset with CER up to 20% |
| Granary | text, language ID and hallucination filters | the paper names no audio quality predictor (searched for DNSMOS, SNR and MOS) |
| LibriTTS-R | speech restoration | restores LibriTTS audio rather than filtering on a score; no quality threshold described (searched) |

- Emilia-Pipe on a 600 h sample raised the mean DNSMOS OVRL from 2.50 (raw) to 3.26 and kept 29.43% of the audio.
- Data-Speech averages Brouhaha's SNR and C50 over frames inside Brouhaha's own speech segments, using a copy of the checkpoint on Hugging Face (`ylacombe/brouhaha-best`).
- Data-Speech bin edges (`v02_bin_edges.json`): SNR 17.1 to 75.0 dB, C50 10 to 60 dB, SI-SDR -17.8 to 34.4 dB, PESQ 1 to 4.5.

## Recommendation

A proposal for the owner, not a sourced fact.

### Per clip, in the manifest

1. Signal checks, own NumPy code, every clip:
   - duration, source sample rate, peak level, DC offset (mean);
   - clipping: share of samples at or within one LSB of full scale, and the longest run of them;
   - digital silence: share of exact zeros and the longest zero run;
   - loudness with `pyloudnorm` (BS.1770-4) and plain RMS level, for level normalisation before mixing;
   - effective bandwidth: the highest frequency where the long-term speech spectrum stays within a fixed margin of the in-band level, which separates upsampled narrowband (CallHome, some Common Voice phones) from wideband.
2. Brouhaha (GitHub checkpoint, MIT): mean SNR and C50 over speech frames, the 10th percentile of frame SNR, and Brouhaha's speech duration as a cross-check against the forced-alignment labels.
3. WADA-SNR from the paper, own code: a second, independent blind SNR at almost no cost; it is also the LibriTTS precedent.
4. Squim objective (CC BY 4.0, already in torchaudio): SI-SDR, PESQ and STOI as a second view on noise and distortion, run on the first 15 s as Data-Speech does.

All four run in PyTorch or NumPy without ONNX Runtime, and none has an NC licence.
Brouhaha probably needs its own uv environment because of its `pyannote.audio` and NumPy pins.

### Analysis only, on a stratified sample

- DNSMOS P.835 on a few thousand clips per source, for comparison with Emilia and the literature; it runs under `benchmarking/` or as an external tool, unless the owner allows ONNX Runtime for data preparation.
  Converting the ONNX files to PyTorch is possible in principle but not tested.
- Audiobox Aesthetics PQ on the same sample: the only predictor with published correlations on Common Voice clips.
- NISQA on the CallHome and telephone-simulated clips only, to check that the colouration dimension flags narrowband audio, since its dimensions map onto channel faults (packet loss, clipping); its weights are NC-SA.
- Skip UTMOS22, UTMOSv2 and Squim subjective: they rate text-to-speech naturalness or need a reference, and none answers a VAD question.
- ITU-T P.808 crowdsourced MOS (`microsoft/P.808`) only if a perceptual quality claim is needed; it is not needed for VAD labels.

### Thresholds

- Drop only broken clips: no speech detected by Brouhaha while the transcript is not empty, digital silence over most of the clip, heavy clipping, or a failed alignment ([0025](../project/adr/0025-alignment-labelling.md)).
- ASR agreement where a check is needed: a high error rate means the transcript does not match the audio, so the alignment and the labels are wrong; CER above 50% means that it does not match (People's Speech); 20% is the stricter VoxPopuli line.
- No MOS threshold: the target is noisy telephone audio, and Emilia's OVRL above 3.0 rule would remove the device conditions OpenVAD wants.
- Calibrate every numeric threshold on a hand-checked sample per source before use; the LibriTTS 20 dB WADA-SNR line is a starting point for a "clean foreground" pool, not a drop rule.

### Using a blind SNR for Common Voice mixing

- Treat the clip's own noise as present: with a baseline SNR $s_0$ and added noise at $s_a$ relative to the same speech power, the effective SNR is $-10 \log_{10}(10^{-s_0/10} + 10^{-s_a/10})$, assuming independent noise.
- Draw the target effective SNR, solve for $s_a$, and skip the mix (keep the clip as is) when the target is above $s_0$.
- Store $s_0$, its estimator and the effective SNR in the manifest, and stratify results by the effective SNR within each dataset.
- Use Brouhaha as the primary $s_0$ and WADA-SNR as the check; where they disagree by more than a few dB, or Brouhaha reports above 30 dB, record the clip as "clean, SNR uncertain" rather than trusting the number.
- Run every estimator on the 16 kHz signal that training sees; compare scores within a bandwidth class, never across 8 kHz and 16 kHz content.
- Time each tool on 10 h of Common Voice on the RTX 4090 before the full run; at 40.27 G MACs per 5 s, Squim alone is about $2.9 \times 10^{13}$ MACs per hour of audio.

## Not verified

- Which licence covers the DNSMOS ONNX files (MIT code or CC BY 4.0 content), and which training stage `model_v8.onnx` is.
- Behaviour of every tool on 8 kHz content upsampled to 16 kHz, and on Common Voice phone recordings.
- Brouhaha SNR accuracy on real recordings and above 30 dB; whether the GitHub checkpoint, `pyannote/brouhaha` and `ylacombe/brouhaha-best` are the same weights; whether Brouhaha installs under Python 3.12 with current `pyannote.audio`.
- GPU and CPU throughput of each tool on its own.
- UTMOS22 weight licence; whether UTMOSv2 inference is deterministic with its crops; that `facebook/wav2vec2-base` was pretrained on English only.
- Accent and language bias of the MOS predictors beyond the published per-category numbers.
- Whether metadata made with NC or SA weights carries obligations.
- National Institute of Standards and Technology STNR licence and current availability.
- Squim objective accuracy on real noisy recordings and non-English speech.
- Audiobox Aesthetics parameter count and speed.
- SCOREQ weight licence; the VERSA licence discrepancy.
- That `dnsmos_local.py` fails with current `librosa`; that converting DNSMOS to PyTorch reproduces its scores.
- torchaudio 2.11.1 compatibility with the pinned `torch==2.14.1`.

## Sources

- DNS-Challenge repository, DNSMOS: <https://github.com/microsoft/DNS-Challenge> (`DNSMOS/README.md`, `DNSMOS/dnsmos_local.py`, `LICENSE`, `LICENSE-CODE`, `README.md`).
- Reddy et al. 2021, "DNSMOS: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors", ICASSP 2021: <https://arxiv.org/abs/2010.15258>.
- Reddy et al. 2022, "DNSMOS P.835", ICASSP 2022: <https://arxiv.org/abs/2110.01763>.
- NISQA repository: <https://github.com/gabrielmittag/NISQA> (`README.md`, `nisqa/NISQA_lib.py`, `config/finetune_nisqa.yaml`, `env.yml`, `weights/LICENSE_model_weights`).
- Mittag et al. 2021, "NISQA: A Deep CNN-Self-Attention Model for Multidimensional Speech Quality Prediction with Crowdsourced Datasets", Interspeech 2021: <https://arxiv.org/abs/2104.09494>.
- torchaudio Squim: <https://github.com/pytorch/audio> (`src/torchaudio/pipelines/_squim_pipeline.py`, `src/torchaudio/models/squim`, tag `v2.11.1`); maintenance plan: <https://github.com/pytorch/audio/issues/3902>.
- Kumar et al. 2023, "TorchAudio-Squim: Reference-less Speech Quality and Intelligibility Measures in TorchAudio", ICASSP 2023: <https://arxiv.org/abs/2304.01448>.
- UTMOS22: <https://github.com/sarulab-speech/UTMOS22>; Saeki et al. 2022, "UTMOS: UTokyo-SaruLab System for VoiceMOS Challenge 2022": <https://arxiv.org/abs/2204.02152>.
- UTMOSv2: <https://github.com/sarulab-speech/UTMOSv2> (`README.md`, `docs/datasets.md`, `docs/inference.md`, `utmosv2/config/fusion_stage3.py`, `utmosv2/utils/_download.py`, `pyproject.toml`); Baba et al. 2024, "The T05 System for The VoiceMOS Challenge 2024": <https://arxiv.org/abs/2409.09305>.
- Audiobox Aesthetics: <https://github.com/facebookresearch/audiobox-aesthetics>; model card metadata: <https://huggingface.co/facebook/audiobox-aesthetics>; Tjandra et al. 2025: <https://arxiv.org/abs/2502.05139>; PyPI: <https://pypi.org/project/audiobox-aesthetics/>.
- Brouhaha: <https://github.com/marianne-m/brouhaha-vad> (`README.md`, `requirements.txt`, `LICENSE`, `.github/workflows/setup.yml`, `models/best/checkpoints`); model metadata: <https://huggingface.co/pyannote/brouhaha>; Lavechin et al. 2023, ASRU: <https://arxiv.org/abs/2210.13248>.
- Kim and Stern 2008, "Robust Signal-to-Noise Ratio Estimation Based on Waveform Amplitude Distribution Analysis", Interspeech 2008: <http://www.cs.cmu.edu/~robust/Papers/KimSternIS08.pdf>.
- `pyloudnorm`: <https://github.com/csteinmetz1/pyloudnorm> (`README.md`, `pyloudnorm/meter.py`).
- DNSMOS Pro: <https://github.com/fcumlin/DNSMOSPro>; SCOREQ: <https://github.com/alessandroragano/scoreq>; VERSA: <https://github.com/wavlab-speech/versa>; `torchmetrics` DNSMOS: <https://github.com/Lightning-AI/torchmetrics> (`src/torchmetrics/functional/audio/dnsmos.py`).
- Data-Speech: <https://github.com/huggingface/dataspeech> (`README.md`, `dataspeech/gpu_enrichments/snr_and_reverb.py`, `squim.py`, `examples/tags_to_annotations/v02_bin_edges.json`).
- Lyth and King 2024, "Natural language guidance of high-fidelity text-to-speech with synthetic annotations" (Parler-TTS): <https://arxiv.org/abs/2402.01912>.
- He et al. 2024, "Emilia: An Extensive, Multilingual, and Diverse Speech Dataset for Large-Scale Speech Generation": <https://arxiv.org/abs/2407.05361>.
- Zen et al. 2019, "LibriTTS": <https://arxiv.org/abs/1904.02882>; Koizumi et al. 2023, "LibriTTS-R": <https://arxiv.org/abs/2305.18802>.
- Kahn et al. 2020, "Libri-Light": <https://arxiv.org/abs/1912.07875>.
- Galvez et al. 2021, "The People's Speech": <https://arxiv.org/abs/2111.09344>.
- Wang et al. 2021, "VoxPopuli": <https://arxiv.org/abs/2101.00390>; repository: <https://github.com/facebookresearch/voxpopuli>.
- Koluguri et al. 2025, "Granary": <https://arxiv.org/abs/2505.13404>.
- Dependency licences: <https://github.com/pyannote/pyannote-audio>, <https://github.com/speechbrain/speechbrain>, <https://github.com/librosa/librosa>.
