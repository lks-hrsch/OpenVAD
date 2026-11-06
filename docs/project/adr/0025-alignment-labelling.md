# 0025: Alignment labelling

Status: Proposed

## Context

Most training speech has transcripts but no timings, so its frame labels come from forced alignment.
[0022](0022-frame-labels.md) says what counts as speech, and [0023](0023-dataset-composition.md) says which label source each dataset uses; this record says how forced-alignment labels are made and checked.
The aligner facts behind it, including models per language, accuracy and licences, are in [`docs/notes/forced-alignment.md`](../../notes/forced-alignment.md).

## Proposal

### Aligner

- Word alignments come from the Montreal Forced Aligner (MFA) with pretrained acoustic models, dictionaries and grapheme-to-phoneme (G2P) models, one per language (owner, 2026-11-03); we do not train our own aligners.
- Some of these models were trained partly on NC data or on GlobalPhone. They are labelling tools and are not shipped ([0024](0024-training-data-licence-policy.md)).
- The labelling runs when the dataset generator needs it.
- Which model labels which language follows the recommendation in [`docs/notes/forced-alignment.md`](../../notes/forced-alignment.md#which-aligner-for-which-language).
- Kazakh has `kazakh_cv` 2.0.0 (`acoustic/kazakh/cv/v2.0.0` in `MontrealCorpusTools/mfa-models`, checked 2026-11-04).
- Japanese and Mandarin transcripts need word segmentation before alignment; how MFA 3 does it is to be checked when the aligner is set up.
- The model cards list corpora, not splits, so the aligners may have seen Common Voice and MLS dev and test audio. They only produce labels; the labels on those pools may be a little more accurate than on train.

### Drop rules

Per language and source, a clip is dropped if:

- its alignment fails or needs a wider beam;
- its per-utterance log-likelihood is in the bottom 5%;
- a word lasts over 1.5 s or a phone over 0.5 s;
- its speaking rate is more than three median absolute deviations from the median;
- its first word starts in the first 20 ms;
- speech covers less than 20% of it.

### Hand check

500 clips are labelled by hand in Praat (owner, 2026-11-04): 125 English, 125 German, 125 from other languages with `*_mfa` models and 125 from other languages with `*_cv` models.

- Clips are sampled across the alignment-score quantiles.
- Each clip gets one TextGrid tier with speech intervals, labelled without seeing the MFA output.
- About 20% of the clips are labelled by two people to measure agreement between labellers, the ceiling for MFA.
- Reported: frame agreement on the 10 ms grid and the share of onsets and offsets within 20 ms and 50 ms.
- The pass threshold is set after a pilot.

The published MFA results are phone boundary errors on other corpora, so they do not replace this check on noisy crowd-sourced clips.

## Consequences

- Praat is under the GNU General Public License. It is a tool only and never part of the release.
- There is no per-language quota, so the thesis claims no hand-checked accuracy per language. Alignment failure rates, computed automatically per language, flag languages for a closer look.

## Open questions

- Are the drop rules owner-decided? No source dates them.
- Do Italian and Dutch, which have no G2P model, use a G2P model trained on the dictionary for out-of-dictionary words, or drop those clips?
- Which models label Croatian, Bosnian, Czech, Serbian, Bulgarian and Ukrainian? [`docs/notes/forced-alignment.md`](../../notes/forced-alignment.md#mfa) lists what exists; none is chosen, and Bosnian has no model of its own.
- Is the legacy `acoustic/arabic.zip` in `mfa-models` usable? [0023](0023-dataset-composition.md) leaves Arabic out as a language without an MFA model; the note reports the archive as unverified ([`docs/notes/forced-alignment.md`](../../notes/forced-alignment.md)).
- The VoxCommunis coverage, checked model by model.
