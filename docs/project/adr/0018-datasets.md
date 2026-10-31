# 0018: Datasets

Status: Open

## Context

Data must be publishable and reproducible, and the released model uses no non-commercial data ([0004](0004-project-scope.md)).

- Usable: LibriSpeech (CC BY 4.0) + public MFA alignments; AVA-Speech as the independent in-the-wild benchmark; MUSAN / Deep Noise Suppression Challenge / DEMAND as noise.
- Caution: VOiCES (re-recorded LibriSpeech, so not independent); LibriVAD (CC BY-NC-SA); Buckeye (non-commercial); AudioSet (YouTube availability decays); Google Speech Commands (utterance-level labels); ESC-50, WHAM!/LibriMix noise (NC).
- Not for training: TIMIT, RATS, AURORA-2 (Linguistic Data Consortium / European Language Resources Association), NTT Multi-Lingual Speech DB (commercial); NIST SRE and Language Recognition Evaluation are evaluation campaigns (Linguistic Data Consortium), so related work only.
- FLEURS-VAD-102 is not released.

## Open questions

- The training and benchmark dataset selection.
- Licences still marked "verify" (ESC-50, WHAM!, McGill TSP, Buckeye, VOiCES).
- Which "Speech Database" was meant in the owner's dataset list.

## References

- LibriVAD 2025 (SSR)
