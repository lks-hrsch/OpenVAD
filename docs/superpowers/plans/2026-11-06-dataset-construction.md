# Dataset Construction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `openvad_train.dataset`, which turns `$OPENVAD_DATA_DIR/datasets/raw` into `train.parquet`, `validation.parquet`, `test.parquet` and `index.parquet`, with pointers to constructed FLAC in `data/` or to unmodified raw files, plus 10 ms labels and metadata.

**Architecture:** Five stages, `index`, `align`, `plan`, `construct`, `publish` and `verify`, each writing its own output and manifest. `plan.parquet` resolves every random draw, so `construct_sample` is a pure function of a plan row and the raw data. As-is sources skip construction and stay in raw. Labels stay as intervals until the last moment, so changing G never reruns MFA.

**Tech Stack:** Python 3.12, numpy, scipy, soundfile, Polars with async Parquet I/O, `datasets`, ffmpeg as a subprocess, MFA 3.4.3, Rust with PyO3 and the `numpy` crate.

**Spec:** `docs/project/spec/dataset-construction.md`. Read it first; this plan does not repeat its tables.

## Global Constraints

- Everything constructed is 16 kHz mono 16-bit FLAC in uncompressed tars under `data/`. Nothing is stored at 8 kHz. As-is sources are never copied.
- `audio_filepath` is always relative to `$OPENVAD_DATA_DIR`.
- All Parquet reads and writes go through `dataset/parquet_io.py`, asynchronous with Polars.
- All 8 and 16 kHz conversion uses the OpenVAD resampler through `openvad-rs` (ADR 0009). Other rates use ffmpeg, with the version recorded.
- AVA-Speech, Fearless Steps, Picovoice and Simsamu never enter this pipeline. LibriSpeech `dev-clean` and `test-clean` are never read directly. LibriVAD val is validation only, and LibriVAD test is test only (ADR 0024).
- Training data and anything built from it stay on the share and the workstation's temporary copies (ADR 0024).
- `seed = blake2(global_seed, plan_version, sample_id)`, never a running sequence.
- Labels: 0 non-speech, 1 speech, 2 masked, one per 10 ms frame.
- G = 200 ms, applied only at rasterisation.
- Tests mirror the tree under `training/tests/openvad_train/dataset/`, keep file names, and never add an `__init__.py` to a tests tree. No real data in tests.
- No em dashes, British spelling, English only. Conventional Commits. Copy the session's commit trailers.
- New source files carry the MPL-2.0 header used by the existing files. No GPL code in the release; ffmpeg, MFA and Kaldi run as tools.
- Run `just lint-py` and the affected tests before each commit; Rust tasks also run `just lint-rs`, `just test-rs` and `just guard-engine`.

## Review Focus

- A raw clip that cannot be decoded (empty or corrupt MP3, truncated file) must not kill a shard: it is logged to `failures.jsonl`, skipped, and reported by `verify`. Test in Task 8.
- Two processes or machines running `construct` on one share must not write the same shard: a lock refuses the second. Test in Task 8.
- `construct` or `publish` against a plan whose `plan_version` differs from existing shards must refuse rather than mix them. Test in Task 8.
- A noise file shorter than the sample, and a mix whose sum exceeds full scale: noise loops seamlessly, and the whole sample scales down so SNR and labels hold. Test in Task 5.
- An empty, digits-only or punctuation-only transcript must drop with a recorded reason instead of failing an MFA batch. Test in Task 14.

---

## File Structure

```text
training/src/openvad_train/dataset/
  __main__.py          CLI: index, align, plan, construct, publish, verify, report, handcheck-sample, handcheck-score
  config.py            BuildConfig and the SLICE and FULL presets; every share and probability of the spec
  schema.py            Clip, Timing, rank, sample_seed, Polars schemas
  parquet_io.py        async Polars read and write helpers
  audio.py             load_audio, the one reader behind every pointer
  labels.py            gap rule and rasterise
  cutting.py           long-clip and conversation chunking
  index.py             adapter registry, reserved speakers, source guard
  sources/<source>.py  clips() and timings() per source
  align.py             MFA batches, TextGrid conversion, drop rules, cap walker
  mfa_models.tsv       language to acoustic model, dictionary, G2P
  handcheck.py         100-clip sample and score
  noise.py             noise categories and pools, babble, SSN
  musan_noise_categories.tsv
  mix.py               padding, SNR, TIR, dither, transients, peak protection, WADA-SNR
  channel.py           room, handset, AGC, clipping, suppression
  codecs.py            ffmpeg round trips, preflight, delay table
  codec_delays.json    measured delay per codec
  g711.py              A-law and mu-law lookup tables
  narrowband.py        train-time and scoring transforms
  plan.py              sample drawing and shard assignment
  construct.py         construct_sample, shard writer
  publish.py           split Parquet files, index.parquet, README.md, manifest.json
  verify.py            dataset checks
  report.py            checkpoint report
training/src/openvad_train/fetch/rirs.py
training/src/openvad_train/fetch/librivad.py (fetch_librivad_val)
inferencing/openvad-core/src/resample.rs, resample_coefficients.rs
training/src/openvad_train/design_resampler.py
tools/montreal-forced-aligner, tools/kaldi (submodules)
```

## Milestone 0: Records

### Task 1: Record the owner's decisions in the ADRs

**Files:**

- Modify: `docs/project/adr/0009`, `0017`, `0018`, `0020`, `0021`, `0022`, `0023`, `0024`, `0025`
- Modify: `docs/notes/datasets.md`, `docs/notes/forced-alignment.md`, `data/README.md`, `docs/shared/acronyms.tex`

**Interfaces:** Produces the ADR text every later task cites.

- [ ] **Step 1: Read `docs/project/adr/README.md` and the spec's "ADR changes" section.** Rewrite each record to the current state. Mark every decision the spec marks owner as `(owner, 2026-11-06)`. Spec proposals stay unmarked. Change no status line.
- [ ] **Step 2: Add `SSN` to `docs/shared/acronyms.tex`** with `short`, `long` and `definition`. Check `FLAC`, `TIR`, `MFA` exist and add any missing one.
- [ ] **Step 3: Update `docs/notes/datasets.md` use matrix and `data/README.md`** for MUSAN, WHAM! `tt`, CallHome at 16 kHz, DEMAND channels, LibriVAD val and test and the new RIR sources. Correct the LibriVAD row: `medium` is every 10th utterance with 54 mixtures each, about 10 h of unique speech, and recompute the affected totals. ADR 0024 is `Accepted`, so its change is marked `(owner, 2026-11-06)`.
- [ ] **Step 4: Verify.** Run `grep -rn "stored at 8 kHz" docs/project/adr data docs/notes`. Expected: no current-state claim left. Run `grep -rnP "\x{2014}" docs/project docs/notes data/README.md`. Expected: no matches.
- [ ] **Step 5: Commit** `docs(adr): record the dataset construction decisions`.

## Milestone 1: Vertical slice

### Task 2: Package skeleton, identifiers and config

**Files:**

- Create: `dataset/__main__.py`, `dataset/schema.py`, `dataset/config.py`, `dataset/parquet_io.py`
- Modify: `training/pyproject.toml`, `uv.lock`, `justfile`
- Test: `training/tests/openvad_train/dataset/schema.py`, `.../config.py`, `.../parquet_io.py`

**Interfaces:**

- Produces:
  - `rank(sample_id: str) -> int`, 64-bit blake2.
  - `sample_seed(global_seed: int, plan_version: str, sample_id: str) -> int`.
  - `@dataclass(frozen=True) Clip(clip_id, source, language, speaker: str | None, split: Literal["train","validation","test"], path: str, offset_s: float, duration_s: float, licence: str, transcript: str | None, reserved: bool = False)`.
  - `@dataclass(frozen=True) Timing(clip_id, kind: Literal["word","turn","frame"], speech_ms: list[tuple[int,int]], mask_ms: list[tuple[int,int]], metrics: dict | None, provenance: str)`.
  - `BuildConfig` with presets `SLICE` and `FULL`. `SLICE`: SWC German, DEMAND, codecs none and G.722. `FULL`: every value in the spec's tables.
  - `dataset_root(cfg) -> Path`: `data_dir()/datasets/processed/<cfg.name>`, reusing `fetch.common.data_dir`.
  - `async read_parquet(path: Path, columns: list[str] | None = None) -> pl.DataFrame`, through `scan_parquet(...).collect_async()`.
  - `async write_parquet(frames: dict[Path, pl.LazyFrame]) -> None`, through `sink_parquet(tmp, lazy=True)` under `pl.collect_all_async`, then `os.replace` for each file.

- [ ] **Step 1: Write the failing tests.**

```python
def test_rank_is_stable():
    assert rank("swc/a") == rank("swc/a")
    assert rank("swc/a") != rank("swc/b")

def test_seed_depends_on_version_and_id_only():
    assert sample_seed(1, "v1", "x") == sample_seed(1, "v1", "x")
    assert sample_seed(1, "v2", "x") != sample_seed(1, "v1", "x")

def test_full_condition_shares_sum_to_one():
    assert sum(FULL.condition_shares.values()) == pytest.approx(1.0)
    assert FULL.condition_shares["clean"] == 0.15
    assert FULL.condition_shares["competing"] == 0.15
    assert FULL.condition_shares["babble"] == pytest.approx(0.7 / 9)

def test_full_snr_grid():
    assert FULL.snr_grid_db == (-5, 0, 5, 10, 15, 20)

def test_parquet_round_trip_keeps_labels(tmp_path):
    p = tmp_path / "a.parquet"
    df = pl.DataFrame({"sample_id": ["a"], "labels": [[0, 1, 2]]}, schema={"sample_id": pl.String, "labels": pl.List(pl.UInt8)})
    asyncio.run(write_parquet({p: df.lazy()}))
    assert asyncio.run(read_parquet(p)).equals(df)

def test_failed_write_leaves_no_file(tmp_path):
    p = tmp_path / "a.parquet"
    bad = pl.LazyFrame({"a": [1]}).select(pl.col("missing"))
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        asyncio.run(write_parquet({p: bad}))
    assert not p.exists()
```

- [ ] **Step 2: Run** `uv run pytest training/tests/openvad_train/dataset/schema.py training/tests/openvad_train/dataset/config.py -v`. Expected: FAIL, import errors.
- [ ] **Step 3: Add `polars`, `soundfile`, `scipy`, `datasets` to `training/pyproject.toml`, run `uv lock`, and implement the four modules.** `__main__.py` is argparse subcommands dispatching to stage functions, following `fetch/__main__.py`. Add recipe `dataset stage *args:` calling `uv run python -m openvad_train.dataset {{stage}} {{args}}`, and list it in `.claude/CLAUDE.md`.
- [ ] **Step 4: Run tests.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(dataset): add identifiers, config and CLI skeleton`.

### Task 3: Labels

**Files:**

- Create: `dataset/labels.py`
- Test: `training/tests/openvad_train/dataset/labels.py`

**Interfaces:**

- Produces:
  - `apply_gap_rule(words_ms: list[tuple[int,int]], gap_ms: int) -> list[tuple[int,int]]`.
  - `rasterise(speech_ms, mask_ms, n_samples: int, shift_ms: int = 0) -> np.ndarray` of `uint8`, length `round(n_samples / 160)`. Intervals are integer ms; float arguments raise `TypeError`.
  - `speech_ratio(words_ms, duration_ms) -> float`, using the word union at G = 0.

- [ ] **Step 1: Write the failing tests.**

```python
def test_gap_shorter_than_g_is_speech():
    assert apply_gap_rule([(100, 300), (450, 600)], 200) == [(100, 600)]

def test_gap_of_exactly_g_is_non_speech():
    assert apply_gap_rule([(100, 300), (500, 600)], 200) == [(100, 300), (500, 600)]

def test_rasterise_grid():
    y = rasterise([(100, 300)], [], 8000)
    assert y.dtype == np.uint8 and len(y) == 50
    assert y[9] == 0 and y[10] == 1 and y[29] == 1 and y[30] == 0

def test_mask_wins_and_any_overlap_masks():
    y = rasterise([(0, 500)], [(25, 26)], 8000)
    assert y[2] == 2 and y[1] == 1

def test_tie_is_non_speech():
    assert rasterise([(5, 10)], [], 320)[0] == 0
    assert rasterise([(4, 10)], [], 320)[0] == 1

def test_shift_moves_labels():
    assert rasterise([(100, 200)], [], 8000, shift_ms=30)[12] == 0
    assert rasterise([(100, 200)], [], 8000, shift_ms=30)[13] == 1

def test_length_follows_samples():
    assert len(rasterise([], [], 16081)) == 101
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Compute on integer ms per frame with `np.add.at` differences; a frame is speech only with strictly more than half its 10 ms covered.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(dataset): rasterise speech intervals to 10 ms labels`.

### Task 4: Index stage and SWC adapter

**Files:**

- Create: `dataset/index.py`, `dataset/sources/swc.py`
- Test: `training/tests/openvad_train/dataset/index.py`, `.../sources/swc.py`

**Interfaces:**

- Consumes: `Clip`, `Timing`, `rank`.
- Produces:
  - Adapter contract: `clips(raw: Path) -> Iterator[Clip]`, and `timings(raw: Path, clips: Iterable[Clip]) -> Iterator[Timing]` for non-MFA sources.
  - `SOURCES: dict[str, module]`.
  - `guard_path(path: str, split: str) -> None`, raising `ForbiddenSource` for benchmark paths, direct LibriSpeech evaluation paths, and LibriVAD val or test in the wrong split.
  - `reserve_speakers(clips, share: float = 0.05) -> set[tuple[str, str, str]]`, keyed (split, language, speaker).
  - `apply_cap_rank(clips, hours: float = 30.0) -> Iterator[Clip]`, walking rank order per speaker per source.
  - `run_index(cfg) -> None`, writing `work_dir(cfg)/index/<source>.parquet` and `index/manifest.json`, where `work_dir(cfg)` is `datasets/processed/work/<cfg.name>/` (every later `work/...` path in this plan sits under it).

- [ ] **Step 1: Write the failing tests** on a synthetic `aligned.swc`, `info.json` and `audiometa.txt` tree in `tmp_path`:

```python
def test_reader_is_in_exactly_one_split(swc_tree):
    clips = list(swc.clips(swc_tree))
    by_reader = defaultdict(set)
    for c in clips:
        by_reader[c.speaker].add(c.split)
    assert all(len(s) == 1 for s in by_reader.values())

def test_unaligned_token_masks_the_gap(swc_tree):
    t = next(swc.timings(swc_tree, swc.clips(swc_tree)))
    assert (300, 450) in t.mask_ms
    assert t.kind == "word"

def test_guard_rejects_benchmark_paths():
    for p in ("raw/ava-speech/a.flac", "raw/simsamu/a.m4a", "raw/librispeech/LibriSpeech/test-clean/a.flac",
              "raw/picovoice/a.wav"):
        with pytest.raises(ForbiddenSource):
            guard_path(p, "test")

def test_guard_keeps_librivad_evaluation_out_of_train():
    guard_path("raw/librivad/generated/LibriSpeech/test-clean/a.wav", "test")
    guard_path("raw/librivad-val/generated/LibriSpeech/dev-clean/a.wav", "validation")
    with pytest.raises(ForbiddenSource):
        guard_path("raw/librivad/generated/LibriSpeech/test-clean/a.wav", "train")
    with pytest.raises(ForbiddenSource):
        guard_path("raw/librivad-val/generated/LibriSpeech/dev-clean/a.wav", "train")

def test_reserve_at_least_one_per_language():
    clips = [clip(speaker=f"s{i}", language="kk") for i in range(6)]
    assert len(reserve_speakers(clips)) == 1

def test_cap_keeps_rank_prefix_under_30_h():
    clips = [clip(id=f"c{i}", speaker="s", duration_s=3600.0) for i in range(40)]
    kept = list(apply_cap_rank(clips))
    assert sum(c.duration_s for c in kept) <= 30 * 3600
    assert {c.clip_id for c in kept} == {c.clip_id for c in sorted(clips, key=lambda c: rank(c.clip_id))[:30]}
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** SWC splits 80/10/10 by reader using `rank(reader)`; articles without a reader go to train; reuse `swc_licences` from `fetch/swc.py`; write through `write_parquet`; call `guard_path(c.path, c.split)` on every clip in `run_index`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(dataset): add the index stage with SWC`.

### Task 5: Acoustic mix primitives

**Files:**

- Create: `dataset/mix.py`, `dataset/noise.py`
- Test: `training/tests/openvad_train/dataset/mix.py`, `.../noise.py`

**Interfaces:**

- Produces:
  - `mix_at_snr(speech, noise, active: np.ndarray, snr_db: float) -> np.ndarray`, scaling noise from speech RMS over active samples and noise RMS over the same span.
  - `place(speech, total_n: int, offset_n: int) -> np.ndarray`.
  - `loop_slice(noise, offset_n: int, n: int) -> np.ndarray`.
  - `dither(n: int, dbfs: float, rng) -> np.ndarray`.
  - `protect_peak(x) -> np.ndarray`, scaling the whole sample only when its peak exceeds 1.0.
  - `overlay_transient(x, event, offset_n, peak_ratio_db) -> np.ndarray`, plus `synth_click(rng)` and `synth_dtmf(rng)`.
  - `padded_length(speech_s: float, target: float = 0.33, edge_s: float = 0.0) -> float`.
  - `wada_snr(x) -> float`.
  - `category_of_demand(scene: str) -> str`, `demand_channels(split: str) -> range`, and `NoisePool.draw(category, split, rng) -> NoiseDraw(file, channel, offset_n)`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_measured_snr_matches_draw():
    speech, noise = sine(), white()
    out = mix_at_snr(speech, noise, np.ones_like(speech, bool), 5.0)
    assert snr_of(speech, out - speech) == pytest.approx(5.0, abs=0.5)

def test_noise_shorter_than_sample_loops():
    assert loop_slice(np.arange(4.0), 3, 6).tolist() == [3, 0, 1, 2, 3, 0]

def test_peak_protection_keeps_snr():
    x = np.array([2.0, -1.0, 0.5])
    y = protect_peak(x)
    assert np.abs(y).max() <= 1.0 and y[0] / y[1] == pytest.approx(-2.0)

def test_padded_length_gives_33_percent():
    assert 3.3 / padded_length(3.3) == pytest.approx(0.33, abs=0.01)

def test_demand_scene_categories():
    assert category_of_demand("DKITCHEN") == "domestic"
    assert category_of_demand("SPSQUARE") == "street"
    assert category_of_demand("TBUS") == "transport"

def test_demand_channels_split_like_librivad():
    assert list(demand_channels("train")) == list(range(1, 13))
    assert list(demand_channels("validation")) == [13, 14]
    assert list(demand_channels("test")) == [15, 16]

def test_pool_draws_only_split_channels(demand_tree):
    pool = NoisePool(demand_tree)
    draws = [pool.draw("domestic", "test", np.random.default_rng(i)) for i in range(50)]
    assert {d.channel for d in draws} <= {15, 16}

def test_wada_snr_orders_clean_above_noisy():
    assert wada_snr(sine()) > wada_snr(sine() + white())
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** `NoisePool` indexes DEMAND here; City, babble, SSN and MUSAN arrive in Task 18. WADA-SNR follows Kim and Stern 2008; note the reference in the docstring.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(dataset): add acoustic mix primitives`.

### Task 6: Plan stage

**Files:**

- Create: `dataset/plan.py`
- Test: `training/tests/openvad_train/dataset/plan.py`

**Interfaces:**

- Consumes: `Clip`, `Timing`, `BuildConfig`, `sample_seed`, `NoisePool`, `speech_ratio`.
- Produces:
  - `draw_sample(clip, timing, cfg) -> dict`, a plan row with exactly the columns of the spec's index, plus `speech_offset_n`, `total_n` and `speech_ms`, `mask_ms`.
  - `assign_shards(rows, bytes_per_hour: float, target_bytes: int = 1_000_000_000) -> None`, ordered by rank within each split, for `augmented` rows only; as-is rows have a null `shard`.
  - `run_plan(cfg, limit: str | None = None) -> None`, writing `work/plan/plan.parquet` and printing hours and bytes per source, split and condition. `limit` is `--limit` on the CLI, a sample count or hours such as `1000h`, taking the lowest ranks.

- [ ] **Step 1: Write the failing tests.**

```python
def test_plan_is_independent_of_other_clips():
    a = draw_sample(clip("x"), timing("x"), SLICE)
    list(draw_sample(clip(f"y{i}"), timing(f"y{i}"), SLICE) for i in range(10))
    assert draw_sample(clip("x"), timing("x"), SLICE) == a

def test_condition_shares_converge():
    rows = [draw_sample(clip(f"c{i}"), timing(f"c{i}"), FULL) for i in range(20000)]
    assert share(rows, "condition", "clean") == pytest.approx(0.15, abs=0.01)

def test_snr_only_from_grid():
    rows = [draw_sample(clip(f"c{i}"), timing(f"c{i}"), FULL) for i in range(2000)]
    assert {r["snr_db"] for r in rows if r["snr_db"] is not None} <= {-5, 0, 5, 10, 15, 20}

def test_clean_padding_is_dither_in_range():
    rows = [draw_sample(clip(f"c{i}"), timing(f"c{i}"), FULL) for i in range(2000)]
    clean = [r for r in rows if r["condition"] == "clean"]
    assert all(-90 <= r["dither_dbfs"] <= -70 and r["snr_db"] is None for r in clean)

def test_only_augmented_rows_get_shards_in_rank_order():
    rows = plan_rows(sources=["swc", "voxpopuli"])
    assign_shards(rows, bytes_per_hour=63e6)
    assert all((r["shard"] is None) == (r["treatment"] == "as_is") for r in rows)
    built = [r for r in rows if r["shard"] is not None and r["split"] == "train"]
    assert [r["rank"] for r in built] == sorted(r["rank"] for r in built)
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Create the row's `numpy` generator only from `sample_seed`. `bytes_per_hour` starts from `cfg.flac_mb_per_hour` (63), and Task 9 replaces it with the measured value.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(dataset): add the plan stage`.

### Task 7: ffmpeg codec round trips

**Files:**

- Create: `dataset/codecs.py`, `dataset/codec_delays.json`
- Test: `training/tests/openvad_train/dataset/codecs.py`

**Interfaces:**

- Produces:
  - `preflight(required: set[str]) -> dict`, returning version and build flags and raising `MissingCodec` naming the absent encoder.
  - `roundtrip(audio: np.ndarray, codec: str, bitrate: int | None) -> np.ndarray`, always the same length as the input, 16 kHz float32, through pipes.
  - `measure_delay(codec: str, bitrate: int | None) -> float`, by cross-correlation on a chirp, in ms.
  - `delay_ms(codec: str) -> float`, reading `codec_delays.json`.
  - `decode_16k(path: str, offset_s: float, duration_s: float) -> np.ndarray`, using soundfile when the file is already 16 kHz and ffmpeg otherwise, with the `DecodeError` Task 8 catches.

- [ ] **Step 1: Write the failing tests**, marked to skip when `ffmpeg` is absent.

```python
def test_g722_roundtrip_keeps_length():
    x = chirp(16000)
    assert len(roundtrip(x, "g722", 64000)) == len(x)

def test_impulse_lands_at_table_delay():
    x = impulse_at(8000, n=16000)
    y = roundtrip(x, "g722", 64000)
    assert abs(np.argmax(np.abs(y)) / 16 - (500 + delay_ms("g722"))) <= 10

def test_preflight_names_missing_encoder(fake_ffmpeg_without_amr):
    with pytest.raises(MissingCodec, match="libopencore_amrnb"):
        preflight({"libopencore_amrnb"})
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Pad the input tail by one codec frame before encoding, and trim or zero-pad to length after decoding. Measure the G.722 delay once and commit it to `codec_delays.json`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(dataset): add ffmpeg codec round trips with a delay table`.

### Task 8: Construct stage

**Files:**

- Create: `dataset/construct.py`, `dataset/audio.py`
- Modify: `training/src/openvad_train/fetch/__main__.py` and `fetch/common.py`, moving `_take_share_lock` to `common.take_share_lock`; keep the fetcher behaviour.
- Test: `training/tests/openvad_train/dataset/construct.py`, `.../audio.py`, `training/tests/openvad_train/fetch/common.py`

**Interfaces:**

- Consumes: `decode_16k`, `roundtrip`, `delay_ms`, mix primitives, `rasterise`, `take_share_lock`.
- Produces:
  - `construct_sample(row: dict, raw: Path) -> tuple[bytes, np.ndarray]`.
  - `write_shard(rows, tar_path: Path, fragment_path: Path, raw: Path) -> ShardResult(n_ok, n_failed, sha256)`.
  - `run_construct(cfg, workers: int = 11) -> None`, for `augmented` rows only.
  - `load_audio(row: dict, data_root: Path | None = None) -> np.ndarray`, float32 at 16 kHz, dispatching on `audio_format`.
  - `class OutOfSpace(Exception)`, `class PlanMismatch(Exception)`.

- [ ] **Step 1: Write the failing tests** on synthetic raw audio.

```python
def test_same_row_same_bytes(row, raw):
    assert construct_sample(row, raw)[0] == construct_sample(row, raw)[0]

def test_labels_match_audio_length(row, raw):
    flac, y = construct_sample(row, raw)
    audio, sr = sf.read(io.BytesIO(flac))
    assert sr == 16000 and len(y) == round(len(audio) / 160)

def test_impulse_survives_padding_and_codec(raw_impulse_row, raw):
    flac, y = construct_sample(raw_impulse_row, raw)
    audio, _ = sf.read(io.BytesIO(flac))
    assert abs(np.argmax(np.abs(audio)) // 160 - np.argmax(y == 1)) <= 1

def test_load_audio_reads_tar_member_by_offset(tmp_path, rows, raw):
    write_shard(rows, tmp_path / "00000.tar", tmp_path / "00000.parquet", raw)
    row = asyncio.run(read_parquet(tmp_path / "00000.parquet")).row(0, named=True)
    flac, _ = construct_sample(rows[0], raw)
    assert np.array_equal(load_audio(row, tmp_path), sf.read(io.BytesIO(flac), dtype="float32")[0])

def test_load_audio_reads_raw_file_with_offset(wav_on_disk):
    row = {"audio_format": "file", "audio_filepath": wav_on_disk.name, "offset": 1.0, "duration": 0.5}
    assert len(load_audio(row, wav_on_disk.parent)) == 8000

def test_load_audio_reads_callhome_row(callhome_parquet):
    row = {"audio_format": "callhome_parquet", "audio_filepath": callhome_parquet.name, "row_in_file": 0, "offset": 0.0, "duration": 0.5}
    assert len(load_audio(row, callhome_parquet.parent)) == 8000

def test_tar_is_uncompressed_and_opens_with_tarfile(tmp_path, rows, raw):
    write_shard(rows, tmp_path / "00000.tar", tmp_path / "00000.parquet", raw)
    assert all(m.name.endswith(".flac") for m in tarfile.open(tmp_path / "00000.tar", "r:").getmembers())

def test_killed_worker_leaves_no_finished_shard(tmp_path, rows, raw, monkeypatch):
    monkeypatch.setattr(construct, "_publish_shard", raise_keyboard_interrupt)
    with pytest.raises(KeyboardInterrupt):
        write_shard(rows, tmp_path / "00000.tar", tmp_path / "00000.parquet", raw)
    assert not (tmp_path / "00000.tar").exists() and not (tmp_path / "00000.parquet").exists()

def test_rerun_skips_finished_shards(cfg):
    run_construct(cfg, workers=1)
    before = shard_mtimes(cfg)
    run_construct(cfg, workers=1)
    assert shard_mtimes(cfg) == before

def test_stops_when_space_is_short(cfg, monkeypatch):
    monkeypatch.setattr(shutil, "disk_usage", low_disk)
    with pytest.raises(OutOfSpace):
        run_construct(cfg)

def test_corrupt_clip_is_logged_and_skipped(rows_with_corrupt_clip, raw, tmp_path):
    r = write_shard(rows_with_corrupt_clip, tmp_path / "s.tar", tmp_path / "s.parquet", raw)
    assert r.n_failed == 1 and "c0" in (tmp_path / "s.failures.jsonl").read_text()

def test_second_process_is_refused(cfg):
    with take_share_lock("construct-" + cfg.name):
        with pytest.raises(SystemExit):
            run_construct(cfg)

def test_plan_version_mismatch_refuses(cfg_built, changed_plan):
    with pytest.raises(PlanMismatch):
        run_construct(changed_plan)
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Write an uncompressed tar with `tarfile`, recording each member's data offset and length into the fragment, written by `write_parquet`. Stage both locally, verify by re-reading, copy to `.tmp` names on the share, write the sidecar `.sha256`, then `os.replace` the fragment last, so a fragment means a finished shard. `load_audio` seeks to `byte_offset` and decodes `byte_length` bytes with soundfile. Check free space against estimated size plus 50 GB before each shard. Shards are processed in rank order.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(dataset): add the construct stage with resumable shards`.

### Task 9: Split files, publish, verify and run the slice

**Files:**

- Create: `dataset/publish.py`, `dataset/verify.py`
- Test: `training/tests/openvad_train/dataset/publish.py`, `.../verify.py`

**Interfaces:**

- Produces:
  - `run_publish(cfg) -> None`, merging fragments and as-is rows into `train.parquet`, `validation.parquet`, `test.parquet`, and writing `index.parquet`, `README.md` and `manifest.json` (reuse `fetch.common.write_manifest`). Marks `state: interim` when planned shards are missing.
  - `set_small(root: Path, hours: float) -> None`, setting `in_small` by rank in `train.parquet` and `index.parquet` only, with LibriVAD false.
  - `run_verify(cfg) -> VerifyReport`, failing on a hash, unresolved pointer, label length or index mismatch, and reporting shares and speech ratios.

- [ ] **Step 1: Write the failing tests.**

```python
def test_load_dataset_gives_splits_with_pointers(built_fixture):
    ds = load_dataset(str(built_fixture))
    assert set(ds) == {"train", "validation", "test"}
    row = ds["train"][0]
    assert not row["audio_filepath"].startswith("/")
    assert len(row["labels"]) == round(len(load_audio(row, data_root_of(built_fixture))) / 160)

def test_as_is_points_into_raw_and_copies_nothing(built_fixture_with_as_is):
    df = asyncio.run(read_parquet(built_fixture_with_as_is / "train.parquet"))
    as_is = df.filter(pl.col("treatment") == "as_is")
    assert as_is["audio_filepath"].str.starts_with("datasets/raw/").all()
    assert tar_members(built_fixture_with_as_is).isdisjoint(set(as_is["sample_id"]))

def test_index_has_each_sample_once(built_fixture):
    ids = asyncio.run(read_parquet(built_fixture / "index.parquet"))["sample_id"]
    assert ids.n_unique() == len(ids)

def test_in_small_null_until_set(built_fixture):
    assert asyncio.run(read_parquet(built_fixture / "index.parquet"))["in_small"].null_count() > 0

def test_small_is_a_subset_and_leaves_out_librivad(built_fixture_with_librivad):
    set_small(built_fixture_with_librivad, hours=1.0)
    train = asyncio.run(read_parquet(built_fixture_with_librivad / "train.parquet"))
    small = train.filter(pl.col("in_small"))
    assert 0 < len(small) < len(train) and not small["source"].eq("librivad").any()

def test_small_cut_keeps_lowest_ranks(built_fixture):
    set_small(built_fixture, hours=1.0)
    train = asyncio.run(read_parquet(built_fixture / "train.parquet"))
    assert train.filter(pl.col("in_small"))["rank"].max() < train.filter(~pl.col("in_small"))["rank"].min()

def test_broken_pointer_fails_verify(built_fixture_missing_raw):
    with pytest.raises(VerifyError):
        run_verify(cfg_for(built_fixture_missing_raw))

def test_verify_catches_label_mismatch(built_fixture_broken):
    with pytest.raises(VerifyError):
        run_verify(cfg_for(built_fixture_broken))

def test_partial_build_is_marked_interim(built_fixture_partial):
    assert read_manifest(built_fixture_partial)["state"] == "interim"

def test_manifest_records_provenance(built_fixture):
    m = read_manifest(built_fixture)
    assert {"plan_version", "global_seed", "commit", "ffmpeg", "resampler_sha256", "delays", "licences"} <= set(m)
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** The README YAML holds `configs` with `config_name: default` and explicit `data_files` naming the three Parquet files, so Hugging Face never tries to read `data/`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Run the slice on real data.** On a machine with the share mounted, run `just dataset index --config slice`, then `plan`, `construct`, `publish` and `verify`, limited to a few hundred samples. Expected: `verify` passes. Record the measured FLAC MB per hour and samples per second per core in `docs/notes/datasets.md`, and set `cfg.flac_mb_per_hour`.
- [ ] **Step 6: Listen and inspect.** Export 20 samples to scratch WAVs and show the owner a label overlay image, outside the repo. Expected: speech aligns with labels by ear and eye. Report anything off rather than tuning silently.
- [ ] **Step 7: Commit** `feat(dataset): publish a Hugging Face loadable dataset and verify it`.

## Milestone 2: Resampler and narrowband

### Task 10: Half-band filter in `openvad-core`

**Files:**

- Create: `training/src/openvad_train/design_resampler.py`, `inferencing/openvad-core/src/resample.rs`, `inferencing/openvad-core/src/resample_coefficients.rs`
- Modify: `inferencing/openvad-core/src/lib.rs`, `justfile` (`just design-resampler`)
- Test: `inferencing/openvad-core/tests/resample.rs`, `training/tests/openvad_train/design_resampler.py`

**Interfaces:**

- Produces: `pub fn decimate_2x(input: &[f32]) -> Vec<f32>`, `pub fn interpolate_2x(input: &[f32]) -> Vec<f32>`, `pub const HALF_BAND: [f32; 63]`, `pub const GROUP_DELAY_16K: usize = 31`, in samples at 16 kHz.

- [ ] **Step 1: Write the failing tests.**

```rust
#[test] fn interpolate_doubles_length() { assert_eq!(interpolate_2x(&[0.0; 100]).len(), 200); }
#[test] fn decimate_halves_length() { assert_eq!(decimate_2x(&[0.0; 200]).len(), 100); }
#[test] fn impulse_response_is_causal_and_delayed() { /* peak at GROUP_DELAY_16K */ }
#[test] fn known_vector_is_bitwise_stable() { /* committed input and expected bits */ }
```

```python
def test_stopband_at_least_70_db():
    w, h = freqz(coefficients(), worN=8192, fs=16000)
    assert 20 * np.log10(np.abs(h[w >= 4500]).max()) <= -70

def test_passband_flat_to_3400_hz():
    w, h = freqz(coefficients(), worN=8192, fs=16000)
    assert np.ptp(20 * np.log10(np.abs(h[w <= 3400]))) <= 0.1

def test_half_band_structure():
    c = coefficients()
    assert len(c) == 63 and np.allclose(c, c[::-1])
```

- [ ] **Step 2: Run** `cargo test -p openvad-core` and the Python test. Expected: FAIL.
- [ ] **Step 3: Implement.** Design with `scipy.signal`, then generate `resample_coefficients.rs` from the script. If 63 taps miss the bounds, adjust the tap count and update these tests and the spec's starting value; the spec allows this. Use a fixed loop with explicit `mul_add` and a single accumulator (ADR 0026), scalar only, and no streaming state yet.
- [ ] **Step 4: Run** `just test-rs`, `just lint-rs` and `just guard-engine`. Expected: PASS.
- [ ] **Step 5: Commit** `feat(core): add the half-band resampler`.

### Task 11: Bindings and AMR-NB leg

**Files:**

- Modify: `inferencing/openvad-rs/src/lib.rs`, `inferencing/openvad-rs/Cargo.toml`, `Cargo.toml`, `Cargo.lock`, `inferencing/openvad-rs/python/.../_native.pyi`, `.../__init__.py`, `dataset/codecs.py`, `dataset/codec_delays.json`
- Test: `inferencing/openvad-rs/tests/lib.py`, `training/tests/openvad_train/dataset/codecs.py`

**Interfaces:**

- Consumes: `decimate_2x`, `interpolate_2x`.
- Produces: Python `decimate_2x(a: NDArray[np.float32]) -> NDArray[np.float32]`, `interpolate_2x(...)`, and `narrow_roundtrip(audio, "amr_nb", bitrate)` in `codecs.py`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_rust_matches_scipy_reference():
    x = np.random.default_rng(0).standard_normal(16000, dtype=np.float32)
    np.testing.assert_allclose(decimate_2x(x), reference_decimate(x), atol=1e-6)

def test_rejects_wrong_dtype():
    with pytest.raises(TypeError):
        decimate_2x(np.zeros(10, dtype=np.float64))

def test_amr_nb_removes_energy_above_4_khz():
    y = narrow_roundtrip(tone(6000), "amr_nb", 12200)
    assert rms(y) < 0.01 * rms(tone(6000))
```

- [ ] **Step 2: Run** `just setup && uv run pytest inferencing/openvad-rs/tests/lib.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement** with `PyReadonlyArray1<f32>` from the `numpy` crate. If the workstation's ffmpeg lacks `libopencore_amrnb`, document the required build in `data/README.md`; if no build can provide it, this hits the AMR gate: drop AMR legs and report.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(rs): expose the resampler on NumPy arrays and add the AMR-NB leg`.

### Task 12: G.711 and the narrowband transforms

**Files:**

- Create: `dataset/g711.py`, `dataset/narrowband.py`
- Test: `training/tests/openvad_train/dataset/g711.py`, `.../narrowband.py`

**Interfaces:**

- Produces:
  - `alaw_roundtrip(x8k) -> np.ndarray`, `mulaw_roundtrip(x8k) -> np.ndarray`.
  - `resample_only(audio) -> np.ndarray`, 16 to 8 to 16 kHz.
  - `telephone(audio, rng) -> np.ndarray`, with A-law at p = 0.9 and μ-law at p = 0.1.
  - `apply_narrowband(audio, band: str, rng, p_narrow: float = 0.35) -> np.ndarray`.
  - `eval_seed(sample_id: str) -> int`.
  - `label_shift_frames() -> int`, from `GROUP_DELAY_16K`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_alaw_matches_standard_table():
    assert alaw_roundtrip(known_input()).tolist() == pytest.approx(ffmpeg_pcm_alaw_reference(), abs=1e-4)

def test_native_is_untouched():
    x = noise()
    assert apply_narrowband(x, "narrow_native", rng(), 1.0) is x

def test_leg_always_processed():
    x = noise()
    assert not np.array_equal(apply_narrowband(x, "narrow_leg", rng(), 0.0), x)

def test_wide_respects_p():
    x = noise()
    assert apply_narrowband(x, "wide", rng(), 0.0) is x
    assert not np.array_equal(apply_narrowband(x, "wide", rng(), 1.0), x)

def test_length_is_preserved():
    assert len(telephone(noise(16001), rng())) == 16001

def test_resample_only_has_no_quantisation_noise():
    x = tone(1000)
    assert snr_of(x, delay_compensated(resample_only(x)) - x) > 60

def test_label_shift_rounds_to_zero_frames():
    assert label_shift_frames() == 0

def test_eval_draw_is_repeatable():
    a = telephone(noise(), np.random.default_rng(eval_seed("x")))
    assert np.array_equal(a, telephone(noise(), np.random.default_rng(eval_seed("x"))))
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** with lookup tables; no dependency beyond numpy and `openvad_rs`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(dataset): add G.711 and the narrowband transforms`.

## Milestone 3: MFA gate

### Task 13: Build MFA from a submodule

**Files:**

- Create: `tools/montreal-forced-aligner`, `tools/kaldi` (submodules), `tools/README.md`
- Modify: `.gitmodules`, `pyproject.toml`, `uv.lock`, `justfile`, `.claude/CLAUDE.md`, `.gitignore`

**Interfaces:**

- Produces: `just build-mfa`, and `uv run --group align mfa version` printing `3.4.3`.

- [ ] **Step 1: Add submodules.** `tools/montreal-forced-aligner` at tag `v3.4.3`. `tools/kaldi` at the upstream commit Kalpy's CI or conda-forge recipe uses; record it in `tools/README.md`.
- [ ] **Step 2: Add the `align` dependency group** with the editable MFA, pinned `kalpy-kaldi==0.10.5` and `pynini`. Keep it out of the default groups.
- [ ] **Step 3: Check the lock.** Run `uv lock` without `KALDI_ROOT`, then `just setup` in a clean clone. Expected: both succeed without building Kaldi. If `uv lock` tries to build `kalpy-kaldi`, take fallback 1: a separate uv project in `tools/mfa/`, and rerun this step.
- [ ] **Step 4: Add `just build-mfa`**, compiling Kaldi as shared libraries into `tools/kaldi/install`, then `KALDI_ROOT=... uv sync --group align`. Document `initdb` and `pg_ctl` as requirements, and check them first with a clear error.
- [ ] **Step 5: Build on the NixOS workstation.** Run `just build-mfa`. Expected: `mfa version` prints `3.4.3`. If Kaldi fails to compile after a reasonable attempt, stop and report to the owner before taking fallback 2 (conda-forge `kaldi` as `KALDI_ROOT`).
- [ ] **Step 6: Align a real clip.** With `german_mfa` on 20 Common Voice German clips copied to local temp. Expected: TextGrids with words; confirm that `baumwelch` and OpenGrm `ngram` were not needed. Record the throughput.
- [ ] **Step 7: Commit** `build(tools): build MFA 3.4.3 from a submodule in the uv workspace`.

### Task 14: Align stage

**Files:**

- Create: `dataset/align.py`, `dataset/mfa_models.tsv`
- Test: `training/tests/openvad_train/dataset/align.py`

**Interfaces:**

- Consumes: `Clip`, `Timing`, `apply_cap_rank`.
- Produces:
  - `prepare_transcript(text: str, language: str) -> str | None`; digits become `<unk>`, and it returns None when no word remains.
  - `make_batches(clips, batch_hours: float) -> Iterator[list[Clip]]`, in rank order with each speaker kept whole in one batch.
  - `read_alignment(textgrid: Path, metrics: dict) -> Timing`.
  - `DropRules.fit(timings) -> DropRules`, `.apply(timing) -> str | None`, returning the reason; `.save(path)`, `.load(path)`.
  - `walk_cap(speaker_clips, align_batch, rules, hours=30.0) -> Iterator[Timing]`.
  - `run_align(cfg, languages: list[str] | None) -> None`, resumable per batch, writing `work/words/<source>/<language>/<batch>.parquet` and `work/dropped.parquet`.

- [ ] **Step 1: Write the failing tests** with a fake `align_batch`; no MFA needed.

```python
def test_empty_or_digit_transcripts_drop_with_reason():
    assert prepare_transcript("...", "en") is None
    assert prepare_transcript("12 34", "en") is None
    assert "<unk>" in prepare_transcript("i have 12 dogs", "en")

def test_speaker_stays_in_one_batch():
    seen = {}
    for i, b in enumerate(make_batches(many_clips(), 1.0)):
        for c in b:
            assert seen.setdefault(c.speaker, i) == i

def test_rule_word_over_1_5_s_drops():
    assert rules().apply(timing(words=[(0, 1600)])) == "word_too_long"

def test_rule_first_word_in_20_ms_drops():
    assert rules().apply(timing(words=[(10, 500)])) == "starts_too_early"

def test_rule_low_coverage_drops():
    assert rules().apply(timing(words=[(0, 100)], duration=1000)) == "low_coverage"

def test_speech_commands_is_exempt_from_rate_and_onset():
    assert rules(source="speech-commands").apply(timing(words=[(10, 500)], rate="fast")) is None

def test_thresholds_are_frozen_after_first_batch(tmp_path):
    r = DropRules.fit(first_batch()); r.save(tmp_path / "r.json")
    assert DropRules.load(tmp_path / "r.json").loglik_floor == r.loglik_floor

def test_cap_aligns_only_until_30_h():
    calls = []
    list(walk_cap(hours_of_clips(50), recording_aligner(calls), never_drop(), hours=30.0))
    assert sum(calls) <= 31

def test_drops_are_refilled_from_rank():
    out = list(walk_cap(hours_of_clips(50), aligner_dropping_first(5), rules_dropping_marked(), hours=30.0))
    assert sum(t.duration_s for t in out) >= 29

def test_finished_batch_is_skipped(cfg, fake_mfa):
    run_align(cfg, ["de"]); run_align(cfg, ["de"])
    assert fake_mfa.calls == batches_of(cfg)
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Run `mfa align` in a subprocess on a local temp copy, with `--use_g2p` where the table has a G2P model and with `--clean`; read `alignment_analysis.csv` for log-likelihood; delete TextGrids after conversion. Fill `mfa_models.tsv` following `docs/notes/forced-alignment.md`, marking unverified rows.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(dataset): add the MFA alignment stage`.

### Task 15: Pilot, lexicon route and hand check

**Files:**

- Create: `dataset/handcheck.py`, `dataset/lexicon.py`
- Modify: `dataset/mfa_models.tsv`, `docs/notes/forced-alignment.md`
- Test: `training/tests/openvad_train/dataset/handcheck.py`, `.../lexicon.py`

**Interfaces:**

- Produces:
  - `build_lexicon(words: Iterable[str], language: str) -> dict[str, str]`, with a rule-based G2P.
  - `sample_handcheck(cfg, per_language: int = 25) -> Path`, writing WAVs and empty TextGrids across score quantiles, without MFA intervals.
  - `score_handcheck(path: Path) -> dict`, with frame agreement and the shares of onsets and offsets within 20 and 50 ms.
  - `QualityBar`, and `apply_failure_rule(language, hours, passed) -> Literal["keep", "drop", "owner"]`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_sample_covers_quantiles(cfg_with_scores):
    picked = sample_handcheck(cfg_with_scores)
    assert count(picked, "de") == 25 and score_spread(picked, "de") > 0.5

def test_sample_never_writes_mfa_intervals(cfg_with_scores):
    tg = next(sample_handcheck(cfg_with_scores).rglob("*.TextGrid"))
    assert intervals_in(tg) == 0

def test_score_exact_match_is_perfect(perfect_handcheck):
    assert score_handcheck(perfect_handcheck)["frame_agreement"] == 1.0

def test_failure_rule():
    assert apply_failure_rule("xx", 100.0, passed=False) == "drop"
    assert apply_failure_rule("ca", 2636.0, passed=False) == "owner"
    assert apply_failure_rule("de", 1199.0, passed=True) == "keep"
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Check Epitran's licence and PyPI status first, and add it only to the `align` group. Support XPF only if Epitran lacks the language.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit the tooling.** `feat(dataset): add the hand check and rule-based lexicon route`.
- [ ] **Step 6: Run the pilot on the workstation.** Every `*_cv` or VoxCommunis language over 200 h (Catalan, Kinyarwanda, Belarusian, Kabyle, Basque, Luganda, Uyghur, Italian, Bashkir, Georgian, Cantonese and others found), plus English, German, Japanese and Mandarin: about 200 clips each. Expected output: per-language table of failure rate, drop share, throughput. Thresholds are set from this distribution.
- [ ] **Step 7: Ask the owner to label the 100 clips.** Write the sample to the share and hand over the path. Do not continue to Task 20 until it comes back. Tasks 16 to 19 may continue meanwhile.
- [ ] **Step 8: Record results** in `docs/notes/forced-alignment.md` and commit `docs(notes): record the MFA pilot`.

## Milestone 4: All adapters

### Task 16: MFA-labelled source adapters

**Files:**

- Create: `dataset/sources/commonvoice.py`, `mls.py`, `fleurs.py`, `voxpopuli.py`, `speechcommands.py`, `dataset/cutting.py`
- Test: mirrored under `training/tests/openvad_train/dataset/sources/`, and `.../cutting.py`

**Interfaces:**

- Consumes: the adapter contract of Task 4; layouts in `data/README.md`.
- Produces: `cut_long(timing, max_speech_s: float = 6.6, gap_ms: int = 200) -> list[Chunk(start_ms, end_ms)]`, `split_over(timing, max_s: float = 30.0) -> list[Chunk]`.

- [ ] **Step 1: Write the failing tests** on synthetic TSVs and folders.

```python
def test_cv_train_is_validated_minus_dev_and_test_speakers(cv_tree):
    clips = list(commonvoice.clips(cv_tree))
    held = {c.speaker for c in clips if c.split != "train"}
    assert not held & {c.speaker for c in clips if c.split == "train"}
    assert any(c.clip_id == "validated_only_clip" for c in clips)

def test_vp_invalid_rows_are_dropped_and_none_is_train_only(vp_tree):
    clips = list(voxpopuli.clips(vp_tree))
    assert not any(c.clip_id == "invalid_row" for c in clips)
    assert {c.split for c in clips if c.speaker is None} == {"train"}

def test_vp_split_by_speaker(vp_tree):
    by = defaultdict(set)
    for c in voxpopuli.clips(vp_tree):
        if c.speaker: by[c.speaker].add(c.split)
    assert all(len(s) == 1 for s in by.values())

def test_mls_keeps_official_splits(mls_tree):
    by_id = {c.clip_id: c.split for c in mls.clips(mls_tree)}
    assert by_id["mls/de/train_clip"] == "train" and by_id["mls/de/dev_clip"] == "validation"

def test_fleurs_has_no_speakers(fleurs_tree):
    assert all(c.speaker is None for c in fleurs.clips(fleurs_tree))

def test_sc_test_is_testing_list_not_separate_set(sc_tree):
    assert not any("speech_commands_test_set" in c.path for c in speechcommands.clips(sc_tree))

def test_sc_background_noise_is_non_speech(sc_tree):
    t = next(t for t in speechcommands.timings(sc_tree, speechcommands.clips(sc_tree)) if "_background_noise_" in t.clip_id)
    assert t.speech_ms == []

def test_cut_long_keeps_speech_under_limit():
    assert all(speech_in(c) <= 6.6 for c in cut_long(fifteen_second_timing()))

def test_cut_only_at_gaps_over_g():
    assert all(boundary_gap(c) > 200 for c in cut_long(fifteen_second_timing())[:-1])
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Reuse `read_tsv` and `CV_LOCALES` from `fetch/commonvoice.py`, and the same pool rule as `split_stats`. Leave the 8 languages without an MFA model out, per ADR 0023.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(dataset): add the MFA-labelled source adapters`.

### Task 17: As-is source adapters

**Files:**

- Create: `dataset/sources/ami.py`, `callhome.py`, `voices.py`, `librivad.py`
- Modify: `dataset/plan.py`, `dataset/construct.py`
- Test: mirrored under `training/tests/openvad_train/dataset/sources/`, `.../plan.py`

**Interfaces:**

- Consumes: `split_over`, the adapter contract.
- Produces: `treatment = "as_is"` plan rows pointing into `datasets/raw`; `band = "narrow_native"` for CallHome; `measure_voices_delay(voices_wav, source_wav) -> float`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_ami_vocal_sounds_are_non_speech(ami_tree):
    t = one_timing(ami_tree)
    assert overlaps_none(t.speech_ms, LAUGHTER)

def test_ami_splits_follow_partition(ami_tree):
    assert split_of(ami_tree, "ES2002a") == "train"

def test_callhome_masks_every_turn_boundary():
    t = callhome.timing_from_turns([("A", 1000, 2000), ("B", 1500, 3000)])
    assert (750, 1250) in t.mask_ms and (1250, 1750) in t.mask_ms
    assert (1750, 2250) in t.mask_ms and (2750, 3250) in t.mask_ms
    assert t.speech_ms == [(1000, 3000)] and t.kind == "turn"

def test_callhome_split_by_call(callhome_tree):
    assert one_split_per_call(callhome.clips(callhome_tree))

def test_callhome_is_narrow_native_and_points_at_its_row(callhome_rows):
    assert {r["band"] for r in callhome_rows} == {"narrow_native"}
    assert all(r["treatment"] == "as_is" and r["audio_format"] == "callhome_parquet" and r["row_in_file"] is not None for r in callhome_rows)

def test_voices_speaker_rules(voices_tree):
    clips = list(voices.clips(voices_tree))
    assert not any(c.speaker in DEV_CLEAN_SPEAKERS for c in clips)
    assert {c.split for c in clips if c.speaker in TRAIN_100_SPEAKERS} == {"train"}

def test_voices_distractors_are_not_used(voices_tree):
    assert not any("distractor" in c.path for c in voices.clips(voices_tree))

def test_librivad_maps_each_split(librivad_tree):
    clips = list(librivad.clips(librivad_tree))
    assert {c.split for c in clips if "train-clean-100" in c.path} == {"train"}
    assert {c.split for c in clips if "dev-clean" in c.path} == {"validation"}
    assert {c.split for c in clips if "test-clean" in c.path} == {"test"}
    assert any("Concat" in c.path for c in clips) and any("Concat" not in c.path for c in clips)

def test_librivad_labels_by_majority(tmp_path):
    # 10 ms is 160 samples; a tie is non-speech.
    assert librivad.frame_labels(save_samplewise_labels(tmp_path, [1] * 100 + [0] * 60)).tolist() == [1]
    assert librivad.frame_labels(save_samplewise_labels(tmp_path, [1] * 80 + [0] * 80)).tolist() == [0]

@pytest.mark.parametrize("source", ["ami", "voxpopuli", "speech-commands", "voices", "librivad"])
def test_as_is_row_reads_the_exact_source_samples(source, as_is_row_of, raw):
    row = as_is_row_of(source)
    assert row["audio_format"] == "file" and row["shard"] is None
    assert np.array_equal(load_audio(row, raw), source_slice(row, raw))
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Before writing `librivad.frame_labels`, read one `.npy` on the share to confirm what values mean speech, and how labels pair with mixture files (`match_labels.py`); adjust the fixture if needed. VOiCES: check whether the devkit ships source audio. If it does not, measure delay against the LibriSpeech copies available and report the speakers left uncovered; drop those clips rather than guessing their labels.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Measure random reads on the share.** Time `load_audio` on 1,000 random VoxPopuli rows and 1,000 Speech Commands rows. Expected: a reads-per-second figure in `docs/notes/datasets.md` for the checkpoint report. If session seeks are too slow for training, report it; do not copy audio without the owner's decision.
- [ ] **Step 6: Commit** `feat(dataset): add the as-is source adapters`.

### Task 17b: LibriVAD validation split

**Files:**

- Modify: `training/src/openvad_train/fetch/librivad.py`, `fetch/__main__.py`, `data/README.md`
- Test: `training/tests/openvad_train/fetch/librivad.py`

**Interfaces:**

- Consumes: the pinned `librivad` setup, `Files/Noises/*_val.wav`, `Files/Forced_alignments/librispeech_alignments/dev-clean`, and fetched `dev-clean`.
- Produces: `fetch_librivad_val()`, registered as `librivad-val`, writing `datasets/raw/librivad-val/generated/{LibriSpeech,LibriSpeechConcat}/dev-clean/...`, labels under `datasets/raw/librivad-val/labels/`, and a manifest with `use` and `counts`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_val_refuses_without_librivad_setup(tmp_data_dir):
    with pytest.raises(RuntimeError, match="librivad"):
        fetch_librivad_val()

def test_val_never_touches_the_librivad_manifest_tree(fake_upstream, tmp_data_dir):
    before = tree_digest(raw_dir("librivad"))
    fetch_librivad_val()
    assert tree_digest(raw_dir("librivad")) == before

def test_val_counts_are_recorded(fake_upstream, tmp_data_dir):
    fetch_librivad_val()
    assert read_manifest(raw_dir("librivad-val"))["counts"]["LibriSpeech/dev-clean"] > 0
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Copy `Scripts/` and `create_LibriVAD.py` to a temporary working directory, with `Files/Noises` and alignments linked read-only, so the pinned tree and its manifest stay unchanged. Convert `dev-clean` to wav there, run the upstream Concat, label and generator steps for `val` at size `medium`, and move the outputs into `librivad-val/`. Read `create_LibriSpeech_wav.py:81` and `create_labels.py:132` first, and any patch to upstream's split list stays in our wrapper.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Generate on the share.** Run `just fetch librivad-val`. Expected: about 27.4 h NonConcat and 35.2 h Concat (paper's Table 5), about 7 GB. If the counts differ by more than rounding, report it.
- [ ] **Step 6: Commit** `feat(fetch): generate the LibriVAD validation split`.

## Milestone 5: Full chain

### Task 18: All noise conditions

**Files:**

- Create: `dataset/musan_noise_categories.tsv`
- Modify: `dataset/noise.py`, `dataset/mix.py`, `dataset/plan.py`, `dataset/construct.py`
- Test: `training/tests/openvad_train/dataset/noise.py`, `.../mix.py`, `.../plan.py`

**Interfaces:**

- Produces:
  - `NoisePool` extended with `city`, using WHAM! `tr`, `cv`, `tt`, and MUSAN by the table.
  - `build_babble(talkers: list[Clip], offsets, n, raw) -> np.ndarray`, summing 6 energy-normalised talkers.
  - `build_ssn(lpc, n, rng) -> np.ndarray`, and `fit_lpc(speech, order: int = 12) -> np.ndarray`.
  - `mix_competing(speech, active, segments: list[Segment(clip_id, start_n, end_n, rir_id, lowpass_hz)], tir_db, raw) -> np.ndarray`.

- [ ] **Step 1: Hand-review MUSAN.** Run a one-time keyword pass over MUSAN noise filenames, review by hand, and commit the table with columns `file`, `category` (one of the 9, or `unused`). Every noise file has one row; a test enforces it against the fetcher's `musan_licences.tsv`.
- [ ] **Step 2: Write the failing tests.**

```python
def test_wham_split_mapping():
    assert wham_split("train") == "tr" and wham_split("validation") == "cv" and wham_split("test") == "tt"

def test_tt_never_reaches_train(rows_train):
    assert not any("/tt/" in p for r in rows_train for p in r["noise_source_ids"])

def test_babble_uses_reserved_same_language(plan_rows_babble):
    assert all(is_reserved(c) and c.language == r["language"] for r, c in talkers_of(plan_rows_babble))

def test_babble_japanese_any_language_half_the_time(plan_rows_ja):
    assert share_same_language(plan_rows_ja) == pytest.approx(0.5, abs=0.05)

def test_small_language_borrows(plan_rows_kk):
    assert any(c.language != "kk" for _, c in talkers_of(plan_rows_kk))

def test_babble_has_six_talkers(plan_rows_babble):
    assert {len(r["noise_source_ids"]) for r in plan_rows_babble} == {6}

def test_reserved_is_never_foreground(all_rows):
    assert not any(is_reserved_speaker(r) for r in all_rows)

def test_competing_rules(plan_rows_competing):
    for r in plan_rows_competing:
        assert 1 <= len(r["segments"]) <= 3
        assert 0.10 <= coverage(r) <= 0.50
        assert 0 <= r["tir_db"] <= 15
        assert all(1000 <= s["lowpass_hz"] <= 4000 and s["rir_id"] for s in r["segments"])

def test_measured_tir_matches():
    assert measured_tir(mix_competing(*fixture())) == pytest.approx(10.0, abs=0.5)

def test_competing_talker_stays_non_speech(row_competing, raw):
    _, y = construct_sample(row_competing, raw)
    assert labels_equal_clean_foreground(y, row_competing)

def test_ssn_follows_the_speaker_envelope():
    lpc = fit_lpc(speech_like(), order=12)
    assert len(lpc) == 13
    assert spectral_distance(build_ssn(lpc, 16000, rng()), speech_like()) < 3.0

def test_dev_and_test_babble_use_their_own_split(plan_rows_babble_test):
    assert all(c.split == r["split"] for r, c in talkers_of(plan_rows_babble_test))
```

- [ ] **Step 3: Run.** Expected: FAIL.
- [ ] **Step 4: Implement.** Common Voice's mix uses WADA-SNR to estimate speech power before scaling, and records `baseline_snr_db`. Add synthetic transients at p = 0.3, peak ratio 0 to −18 dB.
- [ ] **Step 5: Run.** Expected: PASS.
- [ ] **Step 6: Commit** `feat(dataset): add all noise conditions`.

### Task 19: RIRs, channel effects and remaining codecs

**Files:**

- Create: `training/src/openvad_train/fetch/rirs.py`, `dataset/channel.py`
- Modify: `fetch/__main__.py`, `dataset/codecs.py`, `dataset/codec_delays.json`, `dataset/plan.py`, `dataset/construct.py`, `data/README.md`
- Test: `training/tests/openvad_train/fetch/rirs.py`, `training/tests/openvad_train/dataset/channel.py`, `.../plan.py`

**Interfaces:**

- Produces:
  - `fetch_rirs()`, registered as `rirs`, into `datasets/raw/rirs/{but,openslr28}/` with a manifest and licences, downloading BUT ReverbDB and OpenSLR 28.
  - `apply_rir(x, rir) -> np.ndarray`, peak-aligned to zero, same length.
  - `handset(x, params) -> np.ndarray`, `agc(x, params) -> np.ndarray`, `soft_clip(x, drive_db) -> np.ndarray`, `suppress(x, params) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests.**

```python
def test_rir_does_not_shift_onset():
    y = apply_rir(impulse_at(1000), rir_with_predelay(50))
    assert np.argmax(np.abs(y)) == 1000

def test_rir_tail_is_non_speech(row_with_rir, raw):
    audio_tail_labels = tail_labels(*construct_sample(row_with_rir, raw))
    assert (audio_tail_labels == 0).all()

def test_rooms_are_disjoint_between_splits(all_rows):
    assert rooms(all_rows, "train").isdisjoint(rooms(all_rows, "test"))

def test_handset_highpass_removes_low_tones():
    assert rms(handset(tone(50), fixed_params(hp=200, lp=3400))) < 0.1 * rms(tone(50))

def test_agc_reaches_target():
    assert rms_dbfs(agc(tone(1000, amp=0.01), fixed_agc(-20))) == pytest.approx(-20, abs=2)

def test_clip_stays_in_range():
    assert np.abs(soft_clip(tone(1000), 12)).max() <= 1.0

def test_suppress_lowers_stationary_noise():
    assert rms(suppress(white(), default_params())) < rms(white())

def test_chain_probabilities(rows_full):
    assert share(rows_full, "room") == pytest.approx(0.5, abs=0.02)
    assert share(rows_full, "handset") == pytest.approx(0.7, abs=0.02)

def test_codec_shares_and_bands(rows_full):
    assert share_eq(rows_full, "codec", "g722") == pytest.approx(0.30, abs=0.02)
    assert {r["band"] for r in rows_full if r["codec"] == "amr_nb"} == {"narrow_leg"}

def test_as_is_skips_chain(as_is_rows):
    assert all(r["codec"] == "none" and r["rir_id"] is None for r in as_is_rows)

@pytest.mark.parametrize("codec", ["g722", "amr_wb", "opus", "amr_nb"])
def test_each_codec_keeps_length_and_delay(codec):
    y = encode_decode(impulse_at(8000, n=16000), codec)
    assert len(y) == 16000
    assert abs(np.argmax(np.abs(y)) / 16 - (500 + delay_ms(codec))) <= 10
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement.** Follow the `fetch/` conventions: `download`, checksums where published, `write_manifest`, and tests like `fetch/wham.py`'s. Verify the exact URLs, licences and checksums of both RIR sets before pinning them. Measure and commit delays for each codec and bitrate range. Add both sets to `data/README.md` and ADR 0018.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(dataset): add RIRs, channel effects and the remaining codecs`.

## Milestones 6 and 7: Checkpoint and full run

### Task 20: Checkpoint report

**Files:**

- Create: `dataset/report.py`
- Test: `training/tests/openvad_train/dataset/report.py`

**Interfaces:**

- Consumes: `run_verify`, plan statistics, the pilot table.
- Produces: `run_report(cfg) -> Path`, writing a Markdown report with projected hours and bytes per source, split and condition against free space; measured FLAC MB per hour; construction and MFA throughput with a wall-clock projection; quality bar results; dropped languages; and every placeholder value with a one-line reason.

- [ ] **Step 1: Write the failing test.**

```python
def test_report_flags_overflow(cfg, low_disk):
    text = run_report(cfg).read_text()
    assert "exceeds free space" in text and "plan_version" in text

def test_report_lists_every_placeholder(cfg):
    assert all(k in run_report(cfg).read_text() for k in PLACEHOLDER_KEYS)
```

- [ ] **Step 2: Run.** Expected: FAIL.
- [ ] **Step 3: Implement** from the Parquet files; no new dependency.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(dataset): add the checkpoint report`.
- [ ] **Step 6: Build a Small-sized trial** with `--limit 1000h` into `openvad-trial`, and run `verify` and `report`.
- [ ] **Step 7: Stop and give the owner the report.** Do not start Task 21. The owner settles every proposal in the spec, placeholders included, storage, and the 100-clip result. Record the decisions in the ADRs and `config.py`, then commit `docs(adr): record the checkpoint decisions`.

### Task 21: Full run

**Requires the owner's explicit approval from Task 20.** This uses weeks of workstation time and most of the share.

**Files:**

- Modify: `docs/notes/datasets.md`

- [ ] **Step 1: Confirm the Common Voice fetch has finished,** with `manifest.json` verified. Measure the remaining locales and update the notes.
- [ ] **Step 2: Freeze.** Run `just dataset index`, `align`, `plan`. Expected: `plan_version` is recorded; alignment resumes by batch after any interruption.
- [ ] **Step 3: Construct.** Run `just dataset construct`, pinned to the workstation. Expected: either finishes or stops with `OutOfSpace` at a clean prefix.
- [ ] **Step 4: Publish and verify** whatever exists. Expected: `verify` passes; the manifest says interim when the plan is not fully built. Report the state to the owner.
- [ ] **Step 5: After more storage,** rerun `construct`, then `publish` and `verify`. Expected: finished shards are untouched, by `sha256`.
- [ ] **Step 6: Record measured sizes and hours** in `docs/notes/datasets.md`; commit `docs(notes): record the built dataset`.
