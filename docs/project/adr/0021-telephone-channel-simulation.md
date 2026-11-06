# 0021: Telephone channel simulation

Status: Proposed

## Context

[0009](0009-native-16khz-and-one-resampler.md) decides a native 16 kHz model trained with band-limited augmentation along the production narrowband path.
This record holds the channel chain.

- Production has more wideband than narrowband calls; the exact split is unknown (owner, 2026-11-04). It negotiates G.722 at 16 kHz or G.711 at 8 kHz over SIP ([build notes](../../notes/build-and-hardware.md#production-audio)).
- Narrowband calls are G.711 at 8 kHz, almost always A-law, from the owner's analysis of production SIP traces.
  The caller's original path is not visible: a landline, or a mobile call (AMR-NB, AMR-WB or EVS, for example over VoLTE) transcoded to G.711 in the PSTN.
- The production agent upsamples narrowband calls from 8 kHz to 16 kHz before speech recognition.

## Proposal

### Chain per sample

[0023](0023-dataset-composition.md) sets which sources pass through this chain: the augmented ones. As-is sources skip the stored chain.
Everything stored is 16 kHz (owner, 2026-11-06); nothing is stored at 8 kHz. The narrowband step runs at train time.
The probabilities and ranges below are placeholders the owner may correct at the checkpoint of the [dataset construction spec](../spec/dataset-construction.md).

Stored chain:

1. Acoustic mix ([0023](0023-dataset-composition.md)).
2. Room, p = 0.5: a real room impulse response from BUT ReverbDB (CC BY 4.0) or OpenSLR 28 (Apache-2.0) (owner, 2026-11-06). Rooms are split 80/10/10 by hash, and the response is peak-aligned to zero, so it adds no label shift.
3. Handset, p = 0.7: high-pass 100 to 300 Hz, low-pass 3.4 to 7 kHz, two peaks of ±6 dB.
4. AGC, p = 0.5: target −26 to −16 dBFS, attack 5 to 20 ms, release 100 to 500 ms.
5. Soft clipping, p = 0.15: drive 3 to 12 dB.
6. Classical phone noise suppression, p = 0.3: Wiener filter or spectral subtraction with scipy.
7. Stored codec leg, with the shares below. Shares and bitrates are placeholders until the gateway log query has run.

| Codec | Share | Bitrate | Path | `band` |
| --- | --- | --- | --- | --- |
| None | 40% | | | `wide` |
| G.722 | 30% | 64 kbit/s | ffmpeg | `wide` |
| AMR-WB | 15% | 6.6 to 23.85 kbit/s | ffmpeg | `wide` |
| Opus | 5% | 12 to 32 kbit/s | ffmpeg | `wide` |
| AMR-NB | 10% | 4.75 to 12.2 kbit/s | decimate, ffmpeg, interpolate | `narrow_leg` |

- "None" concerns the codec stage only; it is not the "Clean" acoustic condition of [0023](0023-dataset-composition.md).
- Every 8 and 16 kHz conversion uses the OpenVAD resampler (owner, 2026-11-04).
- The `band` column of the sample row says what the train-time step does. CallHome's mirror holds 8 kHz telephone audio in 16 kHz files; it is stored unchanged with `band = narrow_native`.

Train-time narrowband (owner, 2026-11-06):

| `band` | Behaviour |
| --- | --- |
| `wide` | with probability `p_narrow`: decimate, G.711, interpolate |
| `narrow_leg` | always: decimate, G.711, interpolate, because the AMR-NB caller leg always reaches the network through G.711 |
| `narrow_native` | nothing |

- This replaces storing narrowband samples at 8 kHz: it costs little in training, and a filter change does not force a rebuild of the wideband samples.
- `p_narrow` is a training parameter with default 0.35, a placeholder. The gateway log query of the [build notes](../../notes/build-and-hardware.md#open-questions) no longer blocks construction.
- It applies to every `wide` sample, as-is sources included; the `treatment` column lets training exclude them.
- G.711 is A-law in 90% and μ-law in 10% of the draws (proposed; see Open questions).
- Validation and test draw with a seed derived from `sample_id`, so scoring is repeatable. The scoring conditions are in [0020](0020-evaluation-benchmarks.md).
- Decimator and interpolator together delay about 3.9 ms at 63 taps, which rounds to 0 on the 10 ms grid. The transform computes the shift from the coefficients, and a test asserts the result.

### Tools

- G.711 is a numpy lookup table in `training/`. The engine has no user for it, so it stays out of `openvad-core`, which ships in the wheel.
- AMR-NB, AMR-WB, Opus and G.722 come from ffmpeg, run as an external tool: `libopencore-amrnb`, `libvo-amrwbenc`, `libopus` and the native `g722`.
  The ffmpeg build flags are recorded with every generated set.
  The workstation's ffmpeg lacks the two AMR libraries, so construction needs another build and fails early without it.
- Each codec's delay is measured once by cross-correlation and stored as a fixed table; the frame labels use it.
- EVS only through the 3GPP reference code (TS 26.442 or 26.443), fetched at build time and never vendored, and only once its licence is checked.
  Until then EVS is left out, and the other caller legs remain.
- No GNU General Public License code is linked into the release.
  The ITU-T STL runs only from outside the repo.
  Spotify Pedalboard (GNU General Public License 3.0) is not used.

## Consequences

- Each codec's delay is measured once by cross-correlation and stored as a fixed table, which the frame labels use ([0022](0022-frame-labels.md)).
- The generator needs ffmpeg with the listed libraries; the build flags are part of every manifest.
- Stored sizes do not depend on the narrowband share, because everything is stored at 16 kHz.
- The AMR-NB leg is the only stored narrowband path; if AMR does not round-trip through ffmpeg with a measured delay, the AMR legs are dropped and reported.

## Open questions

- The mix: the shares of the stored codecs and `p_narrow`, including G.722 for customer-PBX trunks, but not the G.711 A-law and μ-law split of narrowband calls (next question). They stay placeholders until the gateway log query in the [build notes](../../notes/build-and-hardware.md#open-questions) has run.
- Whether the 90/10 G.711 split holds. "Almost always A-law" supports the direction, but no source gives the numbers; the same query should show PCMA against PCMU.
- Which codecs production has enabled, and the share of mobile callers.
- Whether the wideband group (AMR-WB, Opus) should stay wider than the production path, where only G.722 is wideband over SIP ([build notes](../../notes/build-and-hardware.md#production-audio)).
- Whether EVS is feasible: the licence of the 3GPP reference code is not yet checked (its archive sits behind a bot check).

## Deferred

- Packet loss, with PLC, DTX and CNG (owner, 2026-11-06, for packet loss). DTX and CNG need per-frame flags from the codec bitstream that are unverified through ffmpeg (proposed). The design that was dropped from v1:
  - lossless in 50% of the samples, otherwise a loss rate drawn from {1, 3, 5, 10, 20}%, in bursts (Gilbert-Elliott model), decoded with each codec's own PLC;
  - loss hits mainly the caller leg, the G.711 leg only a little;
  - open there: who provides PLC for the G.711 leg while the ITU-T STL stays outside the repo, and whether ffmpeg exposes each codec's own PLC for AMR-NB, AMR-WB and G.722.
- A real-channel test set recorded through the production SIP path.
- Echo of the bot's prompt for barge-in.
