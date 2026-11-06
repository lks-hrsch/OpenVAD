# Architecture decision records

One decision per record, named `NNNN-short-title.md`.
A record describes the current state. To change a decision, rewrite the record; git keeps the history.

## Status

The status line holds only one of these values:

- `Accepted (YYYY-MM-DD)`: decided by the owner, on that date.
- `Proposed`: current recommendation, not yet decided by the owner.
- `Open`: undecided; the record collects findings and open questions.

A status changes only when the owner decides, in the same change that records the decision.
Never upgrade a status silently.

A `Proposed` or `Open` record may hold single owner decisions. Mark each with `(owner, YYYY-MM-DD)`; anything unmarked is a recommendation.
In an `Accepted` record, mark only decisions made after the status date.

## Numbers

Numbers are never reused, because code and data manifests cite them.
A merged record keeps one existing number. A removed number stays listed here, and live links to it are updated in the same change.
0020, 0023 and 0024 are cited by manifests on the data share and cannot be removed.

Unused numbers: 0007, 0019.

## Sections

In this order; only the first two are required:

- `## Context`: why this needs deciding, and the facts needed to read it.
- `## Decision` (`Accepted`) or `## Proposal` (`Proposed`). `Open` records use `## Findings`, `## Open questions` or both instead.
- `## Consequences`
- `## Open questions`
- `## Deferred`
- `## References`

Inventory (sizes, counts, URLs, checksums) belongs in `docs/notes/` or `data/README.md`, unless a decision rests on it.

Acronyms are defined in [`docs/shared/acronyms.tex`](../../shared/acronyms.tex).
