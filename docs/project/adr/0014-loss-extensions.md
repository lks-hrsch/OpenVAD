# 0014: Loss extensions

Status: Proposed

## Context

The VAD loss is BCE ([0013](0013-bce-loss.md)).
Braun & Tashev 2021 train on VNR with BCE + α·MAE, α = 0.2.

## Proposal

- If VNR is used as a training target, use MAE for VNR, as in Braun & Tashev 2021. VNR as a second output head is outside the current scope.
- Consider weighted BCE or focal loss for class imbalance, and always report the speech prior.

## References

- Braun & Tashev 2021 (VNR, BCE+MAE)
