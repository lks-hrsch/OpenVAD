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

[0023](0023-dataset-composition.md) sets which sources pass through this chain.

1. Acoustic mix ([0023](0023-dataset-composition.md)).
2. Optional reverberation (room impulse response).
3. Handset response.
4. AGC, gain and soft clipping.
5. Classical phone noise suppression (spectral subtraction or Wiener filter), with p ≈ 0.3.
6. One of three channel groups, with no shares set (see Open questions):

| Group | Rate | Channel |
| --- | --- | --- |
| No channel | 16 kHz | none |
| Wideband | 16 kHz | G.722, AMR-WB or Opus; no G.711 stage |
| Narrowband | 8 kHz | caller leg: none (landline), AMR-NB, AMR-WB or EVS; then 16 kHz to 8 kHz with the OpenVAD decimator and G.711 A-law or μ-law |

- "No channel" concerns the codec stage only; it is not the "Clean" acoustic condition of [0023](0023-dataset-composition.md).
- Every 8 and 16 kHz conversion uses the OpenVAD resampler (owner, 2026-11-04).
- Within narrowband, G.711 is A-law in 90% and μ-law in 10% of the samples (proposed; see Open questions).
- Narrowband samples are stored at 8 kHz.
  Their 16 kHz view comes from the OpenVAD interpolator at load time, the same path the engine uses ([0009](0009-native-16khz-and-one-resampler.md)).
- Codec DTX and CNG are switched on and off at random.
- Packet loss:
  - Lossless in 50% of the samples.
  - In the other 50%, the loss rate is drawn from {1, 3, 5, 10, 20}%.
  - Losses come in bursts (Gilbert-Elliott model) and are decoded with each codec's own PLC.
  - Loss hits mainly the caller leg; the G.711 leg gets only a little.

### Tools

- G.711 is our own code in `openvad-core`.
- AMR-NB, AMR-WB, Opus and G.722 come from ffmpeg, run as an external tool: `libopencore-amrnb`, `libvo-amrwbenc`, `libopus` and the native `g722`.
  The ffmpeg build flags are recorded with every generated set.
- EVS only through the 3GPP reference code (TS 26.442 or 26.443), fetched at build time and never vendored, and only once its licence is checked.
  If it cannot be used, EVS is dropped and the other caller legs remain.
- No GNU General Public License code is linked into the release.
  The ITU-T STL runs only from outside the repo.
  Spotify Pedalboard (GNU General Public License 3.0) is not used.

## Consequences

- Each codec's delay is measured once by cross-correlation and stored as a fixed table, which the frame labels use ([0022](0022-frame-labels.md)).
- The generator needs ffmpeg with the listed libraries; the build flags are part of every manifest.
- The rendered size depends on the mix, because narrowband samples are stored at 8 kHz.

## Open questions

- The mix: the shares of no channel, wideband and narrowband, and of the codecs and caller legs inside them, including G.722 for customer-PBX trunks, but not the G.711 A-law and μ-law split of narrowband calls (next question). It stays undecided until the gateway log query in the [build notes](../../notes/build-and-hardware.md#open-questions) has run.
- Whether the 90/10 G.711 split holds. "Almost always A-law" supports the direction, but no source gives the numbers; the same query should show PCMA against PCMU.
- Which codecs production has enabled, and the share of mobile callers.
- Whether the wideband group (AMR-WB, Opus) should stay wider than the production path, where only G.722 is wideband over SIP ([build notes](../../notes/build-and-hardware.md#production-audio)).
- Whether EVS is feasible: the licence of the 3GPP reference code is not yet checked (its archive sits behind a bot check).
- The probabilities and ranges of reverberation, handset response, AGC and soft clipping, which no record sets.
- Where the rate conversion sits for a caller leg at 8 kHz, such as AMR-NB, since the decimator follows the caller leg in the table above.
- Whether G.711 belongs in `openvad-core`, which ships in the wheel, or on the training side.
- Who provides PLC for the G.711 leg while the ITU-T STL stays outside the repo, and whether ffmpeg exposes each codec's own PLC for AMR-NB, AMR-WB and G.722.

## Deferred

- A real-channel test set recorded through the production SIP path.
- Echo of the bot's prompt for barge-in.
