# 0022: Frame labels for the dominant speaker

Status: Proposed

## Context

A voicebot must react to the caller, not to a television, a room full of people or someone talking next to the phone.
The training data mixes clean speech with noise, babble and competing talkers ([0023](0023-dataset-composition.md)) and runs it through simulated telephone channels ([0021](0021-telephone-channel-simulation.md)), so the labels must say what counts as speech after all of that.
Per-source labels are in the table of [0023](0023-dataset-composition.md); the labelling tooling is in [0025](0025-alignment-labelling.md).

## Proposal

### What counts as speech

Only the dominant speaker is speech; babble and a competing talker are non-speech (owner, 2026-11-01).

### Frame classes

Every frame on the 10 ms grid is speech, non-speech or masked.
Masked is a formal third class that the loss ignores (owner, 2026-11-04); [0013](0013-bce-loss.md) notes the BCE loss mask.
It covers:

- SWC stretches that were never aligned, which are not labelled non-speech;
- CallHome's ±0.25 s around each turn boundary ([0023](0023-dataset-composition.md)).

### Gap rule

- Sources with word or phone timings: a gap between two aligned words shorter than G is speech, a longer gap is non-speech (owner, 2026-11-04).
- Turn-based sources (CallHome, Simsamu) use their turns as given (owner, 2026-11-04).
- Proposed G = 200 ms, after Kretzschmar 2023: people notice pauses from about 200 to 250 ms, and labels running from the first to the last word made his model sluggish.
- G is a generator parameter; the owner decides its value. A G ablation is optional.

### Competing talker

Following Foreground VAD (Yang et al. 2026, arXiv 2609.19856):

- one to three competing-talker segments covering 10 to 50% of the utterance, at a TIR of 0 to 15 dB over speech-active frames;
- each with an RIR and a random low-pass between 1 and 4 kHz;
- talkers drawn from speakers disjoint from the dominant-speaker pool;
- tested at 9, 13 and 17 dB; BG-FAR uses a 6 dB margin against the reference without the competing talker.

### Channel effects

- Labels are shifted by the codec delay table of [0021](0021-telephone-channel-simulation.md).
- Frames filled by PLC stay speech.
- Comfort noise (CNG) is non-speech.

## Consequences

- RQ3: the reference VADs detect any speech, and AVA-Speech counts background speech as speech.
  BG-FAR is reported next to the usual metrics, and the task difference is stated wherever results are compared ([0020](0020-evaluation-benchmarks.md)).
- CallHome's turns of the two sides overlap and are used as given; what that means for the dominant-speaker task is open ([0023](0023-dataset-composition.md)).
- Labels depend on G, the delay table and the alignments, so all three are recorded in the generator manifest.

## Open questions

- The value of G. [`docs/notes/forced-alignment.md`](../../notes/forced-alignment.md) states 200 ms, but it is not decided.
- What BG-FAR computes exactly. The only definition is the one in `docs/shared/acronyms.tex`, "false-alarm rate measured on background audio"; how the 6 dB margin enters is not recorded.
- Whether 9, 13 and 17 dB are our test levels or the paper's. No test pool in [0020](0020-evaluation-benchmarks.md) uses them, and 17 dB lies above the training range of 0 to 15 dB.
- Whether the gap rule also applies to LibriVAD's own labels, or they are used as they are.
- Whether the label shift includes the group delay of the OpenVAD decimator and interpolator for narrowband samples, or only codec delay.

## References

- Foreground VAD (Yang et al. 2026, arXiv 2609.19856)
- Kretzschmar 2023
