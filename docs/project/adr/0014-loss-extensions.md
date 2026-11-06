# 0014: Loss extensions

Status: Proposed

## Context

The VAD loss is BCE ([0013](0013-bce-loss.md)).
Braun & Tashev 2021 train on VNR with BCE + α·MAE, α = 0.2.
VNR has no present user: the frame labels come from forced alignment ([0022](0022-frame-labels.md)), and no data record plans a VNR target.

## Proposal

- If VNR is used as a training target, use MAE for VNR, as in Braun & Tashev 2021. VNR as a second output head is outside the current scope.
- Consider weighted BCE or focal loss for class imbalance. The speech prior is reported anyway ([0004](0004-project-scope.md)).

## Open questions

- Is VNR a candidate target at all?
- Does weighted BCE or focal loss have a present user, or is it deferred?

## References

- Braun & Tashev 2021
