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
- The labelling runs when the dataset generator needs it, in batches of clips taken in rank order per speaker, one MFA run per batch, so each speaker's clips stay together for fMLLR. Audio is read from a temporary local copy on the workstation ([0024](0024-training-data-licence-policy.md)).
- MFA is the latest release, v3.4.3 (2026-10-08), built from source from submodules (owner, 2026-11-06):
  - `tools/montreal-forced-aligner` is a git submodule pinned to that tag, never `main`, which is heading to 4.0 and removes the tokenisers;
  - `tools/kaldi` is a git submodule at a pinned commit, compiled as shared libraries into `tools/kaldi/dist` by `just build-mfa`;
  - the editable MFA, `kalpy-kaldi` built against `KALDI_ROOT`, and the packages MFA needs live in the separate uv project `tools/mfa`, not in the root workspace, so `just setup` and `uv sync` work without the submodules (the first fallback below, taken because an editable path source on a submodule that is not checked out breaks `uv sync`); `just mfa <args>` runs it;
  - `initdb` and `pg_ctl` come from a PostgreSQL server with pgvector, a documented requirement; the other build requirements, the pinned versions and the licences are in `tools/README.md`.
- If a later change breaks the build, the next fallback is conda-forge `kaldi` as `KALDI_ROOT` only, which needs a note to the owner.
- Which model labels which language follows the recommendation in [`docs/notes/forced-alignment.md`](../../notes/forced-alignment.md#which-aligner-for-which-language).
- Kazakh has `kazakh_cv` 2.0.0 (`acoustic/kazakh/cv/v2.0.0` in `MontrealCorpusTools/mfa-models`, checked 2026-11-04).
- Japanese and Mandarin transcripts need word segmentation before alignment; they use the MFA 3.4.3 tokenisers. Digits map to `<unk>`.
- One committed table, `dataset/mfa_models.tsv`, holds language, acoustic model, dictionary and G2P model, following the note.
- A language with an acoustic model but no lexicon gets a rule-based lexicon (Epitran or XPF) (owner, 2026-11-06). If it fails the quality bar, the failure rule below applies.
- The model cards list corpora, not splits, so the aligners may have seen Common Voice and MLS dev and test audio. They only produce labels; the labels on those pools may be a little more accurate than on train.

### Drop rules

The rules below are confirmed as written (owner, 2026-11-06).
Per language and source, a clip is dropped if:

- its alignment fails or needs a wider beam;
- its per-utterance log-likelihood is in the bottom 5%;
- a word lasts over 1.5 s or a phone over 0.5 s;
- its speaking rate is more than three median absolute deviations from the median;
- its first word starts in the first 20 ms;
- speech covers less than 20% of it.

Speech Commands is exempt from the speaking-rate and first-20-ms rules (owner, 2026-11-06).
Drop thresholds are frozen per language and source from the first batch, so later batches cannot change earlier decisions.
Out-of-vocabulary words use `--use_g2p` where a G2P model exists; Italian and Dutch have none, so their out-of-vocabulary words become `spn` (owner, 2026-11-06).

### Quality bar and hand check

Before any language enters the full run (owner, 2026-11-06):

- Pilot coverage: a small sample of every language using `*_cv` or VoxCommunis models with over 200 h in the pool, plus English, German, Japanese and Mandarin.
- Automatic bar per language: alignment failure rate and the share of clips removed by the drop rules. The thresholds are set from the pilot distribution and recorded in the manifest.
- Hand check: 100 clips are labelled by hand in Praat, 25 each from English, German, Italian and Catalan. They are sampled across the alignment-score quantiles, and each gets one TextGrid tier with speech intervals, labelled without seeing the MFA output.
- Reported: frame agreement on the 10 ms grid and the share of onsets and offsets within 20 ms and 50 ms.
- Failure rule: a failing language is dropped and recorded automatically. If it has over 500 h in the pool, the owner decides instead.

The published MFA results are phone boundary errors on other corpora, so they do not replace this check on noisy crowd-sourced clips.
The earlier 500-clip hand check (125 each for English, German, other `*_mfa` languages and other `*_cv` languages, with about 20% double-labelled to measure agreement between labellers) is deferred; the word tables keep what its sampling needs.

## Consequences

- Praat is under the GNU General Public License. It is a tool only and never part of the release.
- There is no per-language quota, so the thesis claims no hand-checked accuracy per language beyond the four pilot languages. Alignment failure rates, computed automatically per language, flag languages for a closer look.

## Open questions

- Which models label Croatian, Bosnian, Czech, Serbian, Bulgarian and Ukrainian? [`docs/notes/forced-alignment.md`](../../notes/forced-alignment.md#mfa) lists what exists; none is chosen, and Bosnian has no model of its own.
- Is the legacy `acoustic/arabic.zip` in `mfa-models` usable? [0023](0023-dataset-composition.md) leaves Arabic out as a language without an MFA model; the note reports the archive as unverified ([`docs/notes/forced-alignment.md`](../../notes/forced-alignment.md)).
- The VoxCommunis coverage, checked model by model.
