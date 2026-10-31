# 0010: Signal representation

Status: Proposed

## Context

All common frontends share four steps: auditory-scale filterbank → envelope + ~10 ms framing → compression → optional discrete cosine transform.

- Zhao & Wang 2013: GFCC's noise robustness comes from the cube-root compression, not the gammatone filters. MFCC with cube root beats GFCC (28.5 vs. 16.4 % SID at 0 dB).
- LEAF: mel + PCEN already reaches 76.4 vs. 76.9 % for full LEAF.
- Anderson et al. 2023, on VAD: learnable filters barely move; static log-mel has the best AUC (0.881).
- Meng et al. 2024: only PCEN learns.
- EfficientLEAF: no clear advantage over mel, LEAF ~340× slower.
- Su et al. 2026: MFCC vs. FBANK is a small, metric-dependent difference for small VADs.
- Mohamed et al. 2012 / Purwins et al. 2019: no discrete cosine transform before neural nets.

## Proposal

- Fixed frontend (STFT magnitude or log-mel) as the baseline.
- Ablations: (1) compression log vs. cube root vs. PCEN (PCEN is causal, cheap, suits telephone level variation); (2) mel vs. equivalent rectangular bandwidth (gammatone) weighting in the FFT domain (same cost); (3) MFCC vs. FBANK (discrete cosine transform on/off).
- LEAF and SincNet are only cited, not implemented in Rust. Both use symmetric (non-causal) filters.

## References

- Davis & Mermelstein 1980
- Mohamed et al. 2012
- Shao, Srinivasan & Wang 2007 and Shao & Wang 2008 (GFCC)
- Zhao & Wang 2013
- Valero & Alías 2012
- Wang et al. 2017 (PCEN)
- Zeghidour et al. 2021 (LEAF)
- Schlüter & Gutenbrunner 2022 (EfficientLEAF)
- Anderson, Kinnunen & Harte 2023
- Meng et al. 2024
- Ravanelli & Bengio 2018 (SincNet)
- Su et al. 2026
- Purwins et al. 2019
