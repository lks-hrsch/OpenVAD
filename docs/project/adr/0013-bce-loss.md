# 0013: BCE loss

Status: Accepted (2026-10-30)

## Context

The VAD target is a per-frame label.
Besides speech and non-speech, a frame can be masked, for example where an alignment is missing ([0022](0022-frame-labels.md)).

## Decision

- BCE is the VAD loss.
- Masked frames are excluded from the loss by a BCE loss mask (owner, 2026-11-04).

## Consequences

- Extensions of the loss are proposed separately in [0014](0014-loss-extensions.md).

## Open questions

- Is there a rationale for BCE over other per-frame losses to record?
