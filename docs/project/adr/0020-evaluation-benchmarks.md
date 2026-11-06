# 0020: Evaluation benchmarks

Status: Proposed

## Context

RQ3 compares OpenVAD with the reference VADs ([0003](0003-isolated-benchmark-adapters.md)) on our own test set and an independent real-world benchmark ([0004](0004-project-scope.md)).
The [benchmark prediction spec](../spec/benchmark-predictions.md) feeds every system the same fixed chunks; each adapter buffers to its model's native call size, and outputs map onto the 10 ms grid by causal hold.
pyannote is the exception: non-causal, labelled as such, an upper bound only.

## Proposal

### Benchmarks

Five benchmarks, each measuring something the others do not:

| Benchmark | Measures for RQ3 | Audio | Labels | Metrics |
| --- | --- | --- | --- | --- |
| AVA-Speech (151 of the 160 labelled videos, each cut to its labelled 15-minute span) | real-world generalisation: film audio with music and noise, never trained on | 44.1 or 48 kHz stereo, downmixed and resampled offline to 16 kHz with ffmpeg, version recorded in the manifest (owner, 2026-11-04) | human, segments in 4 classes; frame speech if any speech class covers most of it | frame AUC and AP, F1 at the dev threshold, miss and false-alarm rates, per class |
| Fearless Steps SAD (Phase 4 corpus) | real narrowband telephone-like channels with human labels | 8 kHz; upsampled 2:1 with the OpenVAD interpolator, so it exists only in the 8 kHz condition | human speech segments | official DCF = 0.75 P_miss + 0.25 P_FA with the official collar, plus the frame metrics above |
| LibriVAD `medium` test, with `small` test as a paper-comparison row, never trained on (owner, 2026-11-04) | controlled noise type and SNR, seen and unseen noise | 16 kHz | forced alignment, per sample, mapped to 10 ms by majority | frame AUC, EER and minDCF as in the LibriVAD paper, per noise, SNR and form (concatenated or not) |
| Picovoice voice-activity-benchmark | comparison with published ROC curves at 0 dB | 16 kHz, rebuilt with the pinned upstream mixer | energy-based on the clean speech, 32 ms frames, unknown frames ignored | ROC (true and false positive rate per threshold) on the upstream 32 ms frames, AUC |
| Simsamu (61 simulated French emergency calls, [0018](0018-datasets.md)) | real telephone channel in the target domain: French conversations with natural turn-taking | 8 kHz AAC, decoded and upsampled 2:1 with the OpenVAD interpolator, so it exists only in the 8 kHz condition | RTTM speaker turns, pauses inside a turn count as speech | frame metrics above with a ±0.25 s collar around every turn boundary (not scored), and without a collar (owner, 2026-11-01); per role, since only the caller side is a telephone channel |

Published Picovoice curves are cited only; Cobra is commercial and is not run.

### Own test set

- Test-broad: the test split of every [0023](0023-dataset-composition.md) source, with conditions drawn like train, reported per dataset (owner, 2026-11-03). Its size follows from the 0023 splits. It gives the headline numbers. Its DEMAND noise comes from channels 15 and 16, and its LibriVAD part is `medium` test as generated (owner, 2026-11-06). Its City noise is WHAM! `tt` (proposed, as in LibriVAD).
- Test-grid: a paired factorial grid on LibriSpeech `test-clean` and SWC German test, with 19 acoustic × 7 channel cells plus 32 packet-loss cells (165 cells), about 60 utterances per language in every cell, about 130 h. It shows where each system breaks.
- Test-unseen-noise: the same draws as Test-broad with WHAM! `tt` in place of the ambient pool. `tt` is this pool's noise (owner, 2026-11-03). Since `tt` is also City noise in Test-broad, its definition is revisited when it is built (deferred).
- Metrics: frame AUC and AP, F1 at the dev threshold, miss and false-alarm rates, BG-FAR ([0022](0022-frame-labels.md)) and DCF (0.75 P_miss + 0.25 P_FA, as for Fearless Steps). BG-FAR needs the reference without the competing talker, so it is available only on samples where we mixed one (owner, 2026-11-06).
- LibriSpeech `test-clean` is benchmark audio only and never enters dataset creation ([0023](0023-dataset-composition.md)). Its labels are the Zenodo alignments (owner, 2026-11-04).

### Conditions

Three conditions over validation and test (owner, 2026-11-06):

- 16 kHz: the stored audio.
- 8 kHz: derived from the same test set with the OpenVAD resampler (decimate, interpolate; no G.711), so bandwidth is the only variable.
- Telephone: decimate, G.711, interpolate, a labelled third condition for the target domain. The narrowband step is applied at scoring time, because nothing is stored at 8 kHz ([0021](0021-telephone-channel-simulation.md)).

Samples whose `band` is `narrow_leg` or `narrow_native` stay in every condition and are reported through `band`.

### Scoring

- Every system, reference VADs included, is scored on its raw `speech_prob` against the threshold from dev. Post-processing is not part of the benchmark (owner, 2026-11-04). Where post-processing runs is open in [0005](0005-pipeline-stages.md).
- Thresholds (owner, 2026-11-01): one threshold for all benchmarks, chosen on our own dev split ([0023](0023-dataset-composition.md), never trained on); headline F1, miss and false-alarm rates use it. Fearless Steps additionally reports a labelled second operating point tuned on its official Dev split and applied to Eval. AUC, ROC and the official DCF need no threshold.
- OpenVAD labels only the dominant speaker as speech, while the reference VADs and AVA-Speech count any speech ([0022](0022-frame-labels.md)). The task difference is stated wherever results are compared.
- Bootstrap confidence intervals over files and the speech prior are reported for every benchmark ([0004](0004-project-scope.md)).

### Reporting

- Results are per dataset only: no per-language results and no per-family headline (owner, 2026-11-04).
- There is no macro average over English, German and French (owner, 2026-11-03).
- There is no unseen-language test; the thesis claims nothing about unseen languages (owner, 2026-11-04).

### Noise in the test audio

- The Picovoice benchmark mixes whole files from all 16 channels of 10 DEMAND environments (DKITCHEN, DWASHING, NFIELD, NPARK, NRIVER, OHALLWAY, OOFFICE, PSTATION, STRAFFIC, TCAR) into LibriSpeech `test-clean`. Its speech is held out, but DEMAND is training noise with no hold-out (owner, 2026-11-01), so Picovoice results are reported as seen noise.
- LibriVAD's City noise is cut from WHAM! by split: train from `tr`, validation from `cv`, test from `tt` ([0018](0018-datasets.md)). WHAM! `tt` never enters training, so the City test noise is unseen; `tr` is training noise ([0023](0023-dataset-composition.md)).
- The LibriVAD paper calls Babble, SSN and Domestic noise unseen for its models. OpenVAD trains on all nine categories, so for OpenVAD they are seen noise, and this is stated wherever LibriVAD results are reported.

## Consequences

- Fearless Steps and Simsamu add a third kind of condition: native narrowband audio, not derived from the 16 kHz test set.
- Simsamu is small (61 recordings, 14 speakers) and simulated, so its bootstrap confidence intervals are wide; its turn labels favour systems that bridge short pauses.
- AVA-Speech results use 151 videos and cannot be compared one to one with papers that used all 160.
- LibriVAD paper numbers use the `small` test set, which is generated as well; `medium` test-clean is a superset of its speech files but mixes different noise segments.

## Open questions

- What the 19 acoustic, 7 channel and 32 packet-loss cells of Test-grid are, and how the channel cells map to the groups of [0021](0021-telephone-channel-simulation.md). Neither the tree nor git history defines them.
- Which Fearless Steps splits have public reference labels, and the official collar for Phase 4.
- How 10 ms outputs are scored on Picovoice's 32 ms frames.
- Whether "one threshold" means one per system, and whether the 8 kHz condition gets its own dev threshold or reuses the 16 kHz one.
- Whether the DEMAND environment list above matches upstream Picovoice; the fetcher does not confirm it.

## Deferred

- The AVA-Speech 16 kHz step: the fetcher only cuts with ffmpeg and keeps the source rate, so the resampling step does not exist yet and must record its own ffmpeg version.
- Running the 8 kHz and telephone conditions (need the OpenVAD resampler and the G.711 transform).
- The scoring module and the reference VAD adapters.
- Onset-latency evaluation under noisy labels (S4VAD 2026).

## References

- AVA-Speech 2018, kiloVAD 2026, S4VAD 2026, LibriVAD 2025
- Picovoice voice-activity-benchmark, commit `38351b25ba883d4ae414bd48e16e8d4999813edf`
- Fearless Steps Challenge Phase 4, scoring toolkit `aditya-joglekar/fs02_scoring_toolkit`
