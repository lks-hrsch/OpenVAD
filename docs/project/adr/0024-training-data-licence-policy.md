# 0024: Training data licence policy

Status: Accepted (2026-11-03)

## Context

Public speech and noise corpora come under many licences: CC0, CC BY, non-commercial (NC), ShareAlike (SA), NoDerivatives (ND), GFDL-only, or none at all.
The project needs one rule for what may be trained on, what is recorded, and where copies of the data may live.

## Decision

1. The licence of a training dataset does not restrict its use for training.
   NC, SA, ND, GFDL-only and unlicensed audio may all be used.
   - Basis: the owner's decision. The reasoning given was that the weights of a VAD do not reveal its training data.
   - This is not a finding of the university legal office, and this record claims nothing beyond the owner's decision and that reasoning.
2. These rules hold:
   - Public data only: anyone can obtain it, and it needs no paid licence (no Linguistic Data Consortium or European Language Resources Association corpora). Free account gates and accepted terms count as public, as long as access needs neither payment nor individual approval (owner, 2026-11-04).
   - Datasets and third-party weights are never redistributed; scripts fetch the data (owner, 2026-11-04). OpenVAD's own weights are published ([0004](0004-project-scope.md)) under the Mozilla Public License 2.0 ([0027](0027-project-licence.md)) (owner, 2026-11-06).
   - Reproducibility is the claim of the thesis, not licence purity.
   - Labelling tools under an NC licence are acceptable; they are not shipped. MFA models trained partly on GlobalPhone (European Language Resources Association) are labelling tools, not training data, so the public-data rule does not exclude them ([0025](0025-alignment-labelling.md)).
3. Every manifest records the licence of each source, and per file where it varies.
   The release notes list, per source the weights saw, its licence and any terms of use accepted to obtain it (the Hugging Face gate on `talkbank/callhome`, the TalkBank ground rules, the Mozilla Data Collective terms), so anyone who deploys them can judge for themselves.
   On the Mozilla Data Collective, a Claude Code subagent ticked the agreement of 39 Common Voice locales at the owner's request (2026-11-04); 10 were accepted before that, and the owner accepted Korean, Vietnamese, Kazakh and Hausa by hand.
   This is metadata, not a filter.
4. One model, no licence variants.
   A second model variant by licence was considered and not taken.
5. Benchmark data never enters training, whatever its licence:
   - AVA-Speech;
   - Fearless Steps, all splits;
   - the Picovoice voice-activity-benchmark;
   - Simsamu;
   - the LibriVAD `medium` test split;
   - the LibriVAD `small` test split (owner, 2026-11-04);
   - every test pool of [0020](0020-evaluation-benchmarks.md).

   Dev pools are not trained on either.
6. Every training dataset and anything rendered from them stay on the owner's network share (owner, 2026-11-04).
   - The workstation may read them over the mount and hold temporary local copies for rendering and training.
   - Copies anywhere else, including the TU Dresden HPC and the company cluster, need a new decision by the owner.
   - The permissions on the share stay as they are.

## Consequences

- [0004](0004-project-scope.md) points here for which data may be trained on.
- Licence facts per source are recorded in [`data/README.md`](../../../data/README.md) and the manifests; [0018](0018-datasets.md) does not sort sources by licence.
- One run set for RQ1.
- A source's licence cannot keep it out of training; only decision 5 and the composition of [0023](0023-dataset-composition.md) can.
- Decision 6 blocks training on TU Dresden Capella and the company cluster until the owner decides otherwise.

## Open questions

- Whether trained weights count as adapted material under CC BY-NC-SA is legally unsettled. Ask the university legal office before release.
- Whether anything built from LibriVAD (generated audio, labels, checkpoints) must be marked NC in the repo docs and the manifests.
- Whether Fearless Steps registration and the TalkBank login meet decision 2 (neither payment nor individual approval). Their access terms have not been checked.
