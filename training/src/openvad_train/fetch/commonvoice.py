# Copyright (c) 2026 Lukas Hirsch
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Common Voice Scripted Speech 27.0 through the Mozilla Data Collective API: the locales with an MFA model."""

import collections
import csv
import datetime as dt
import json
import os
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from openvad_train.fetch.common import (
    TIMEOUT_S,
    _check,
    cache_dir,
    download_ranges,
    raw_dir,
    tar_filter,
    write_manifest,
)


CV_RELEASE = "cv-corpus-27.0-2026-09-11"
CV_API = "https://mozilladatacollective.com/api"
# MDC dataset ID and archive SHA-256 per locale; the checksums come from the release stats in
# github.com/common-voice/cv-dataset at commit 65ca2460d5e686ea03902a932bbca09b322dc4b0. Locales with a pretrained
# MFA model (ADR 0023), without Croatian (0 h).
CV_LOCALES = {
    "en": ("cmu5jplf300nwmh07iqvk9leo", "2fbc16e9811e0a727c7fde5c71158a9295a2aa5e26f9ec21b305f0a8368a10c7"),
    "de": ("cmu5ktqhe00pgmh07p1xamabd", "74b8a87761afc4e1a9feca8ff0447b11106b2e13959ee8d78d01665945be011d"),
    "fr": ("cmu5kub4600pkmh078n7n5ogl", "40d30f8b173367597efdb8997a73852e6ae09882aee005094a5c583757f3d96c"),
    "ca": ("cmu5k42ge00ojmh07dqv6wzk8", "84ec56b9db9d2fffe32501deb458b7da5169302022c3caf05518747663055b87"),
    "rw": ("cmu5k4i0f00p2mi0771dl1mg8", "aa619505b1b3f0aab883aeafa2ec2a89e0c970b8616c865ca2080cfd7ea8534d"),
    "be": ("cmu5kjjuj00pomi07klnk26oe", "ce73ba8396098d46816657b1222a8a72d9cfffe6aa64fe0a1ea63dbf82c2325b"),
    "es": ("cmu5kitx700pemi073feoiqqu", "b7c1f303ebe3c4a637afdd3d31c5aaa3da131a893ce77169db182068f342e4c1"),
    "kab": ("cmu5xcvp800eao107zfex7alc", "5b9e92714553f8f4b08a50f9cc14ad8e648a07ccf22e2e3c4193687ebee6867c"),
    "eu": ("cmu5xc07l00e6o10740qugb6a", "3ee9ee7b5016e9adf7401e24c6582159c8975551e5c7dcd78c8f132452b965ba"),
    "ug": ("cmu5x7x6w00duo107puutvcvu", "37879c5b12be4ad7006128e8c2d1ab57b8b8184bbdf5ed24e5fc91aec02f551a"),
    "lg": ("cmu5x8jne00f6nq07v8g558pv", "163a4b4b121459c09ccc10468afb139eda6458ecd2c9698d09f41c7a79fbd07b"),
    "sw": ("cmu5sgles002zo1071nphgtby", "45811dfd9c6146513e6ef350c5d60b7bc97ce55411e553e204ac082b31bdc2ba"),
    "ja": ("cmu5xbhnf00fmnq07z9p3aa70", "dba2e11b7b1365360a5f1f31a97cd83410760b9071734c0376d896428c299882"),
    "it": ("cmu5xagm300dyo1075qdhf20m", "27e1ef0e2edfde2a5b0f6557fe6ad8c2c611e9508522be64c4fb93dec0e7c850"),
    "ru": ("cmu5x45pn00dao107j4o9w2yv", "3f269caf4757a7837048153067ca84e29c53fd5dee9394a55e6ecc92afb6055a"),
    "ba": ("cmu62jhk700nwo107l41a2fch", "f806614d5bf1974cd4cbd59191ff0bd198388075bed7010537b7849166b44095"),
    "ka": ("cmu5x9qda00fenq07byl8rlc8", "8099ad5a341466ed9993fc3c5984a65cdacdfdf51407b9474c06e5c5da2f1fb6"),
    "zh-CN": ("cmu5lcunh00qimh078sdd5so8", "e3aeb837ad5148b9081ee3cdda00f8f1021ac30ee261a6527525b721347ed7d8"),
    "yue": ("cmu5x3i5w00d6o107yvwox9lg", "ba060ea5326a43467741a826602518b833a0dea76068f9e40b9f0cc730364bfc"),
    "ta": ("cmu5x7cbc00dqo1071wdidlpf", "b034c57afc4ae01b4f77fe5c6bff4f2084338e1f6ce714da0ee6e83612b66b98"),
    "pt": ("cmu5wui2w00e8nq07idithpm6", "d5c6e70d669378ce8178b2a0f03e3281d67c4ee5ddd5200718221b28e0b93e60"),
    "ab": ("cmu5wtv3c00e4nq07ojsfyprl", "7ba5aa8e08e2cc5b7aec14a973a41800ee5d84cb2680cf5308c3f33d0844c10e"),
    "pl": ("cmu5wsxwp00e0nq07m8v0274j", "c9071ed61cf1836ccabf03a9a9cfeab827f8b7b4342b02a809dceb7597303e57"),
    "th": ("cmu5x5tj000f2nq0755h8qcy5", "f049ca3a3c2e8ce4bfc09dfcff9135a4f9d5678a15e4ad0873d71c90fd39f4f8"),
    "cy": ("cmu5wruey00cio107ofcdoq4u", "17a62f6e4ee29cf40b66bae6ac192fc96344de01c88cb9c047e7d7c413307a9b"),
    "tr": ("cmu5wkah500c2o10719lem8m8", "10becdc2fb112f8cf7ffd983e269dd2013aef5cb3f747b7e5f455146a21d6c8e"),
    "hu": ("cmu5wq4u200dknq072xswzofb", "0912c72c5ecbd0bfb4a8721ed83b7f31626ea8dedb4c4fbc7319bce815d5f0e6"),
    "nl": ("cmu62gjbl00nso107n7ct8dyl", "d6ea62f46bc2f56e15d6155a1eda07b48bcec84062a96d34617ff8e5be8e4763"),
    "zh-HK": ("cmu5wp2bp00dgnq07ylmgbnrz", "4e2deec11bd06e9401675dc1bc1d082f591296571f47a6af3f750f626848dc0e"),
    "uk": ("cmu5wgebl00d0nq07ktip1i6c", "5ae4f8f708adc2718366106161b24e7e969cc1cf87e0367013f076170e4e00b8"),
    "uz": ("cmu43jd6u0010mi07kl431xrw", "02455347111da1d17ccec0e09bbb9917262ee829d6590d6181ee0e0f4f58cd5f"),
    "zh-TW": ("cmu5wkw5i00c6o107jnm3t51u", "96615945c476f795e03b666d1793b5e7fa8f080bfe92cb73c54fcf6254e2b094"),
    "kmr": ("cmu5wfe5300boo107fblbekt9", "163d3165feebba2384a2c2f3e83cada7c1654efe6a87c899bc035ae6806e093e"),
    "cs": ("cmu5wz04e00eenq07xy4d6rhz", "31691c303c70c38a42f8f8434d3819c1d91b3d2e0abbc7996d42cfc3de457afe"),
    "et": ("cmu5wd9o900conq07dt8090ll", "f24eb17517a0bd1f1f5dd91dee049913893c7fcde948546892ea817153161217"),
    "mn": ("cmu5z0n5x00fdo10709hn7u8m", "05c5a447be5136f08cfd7eb9b3c0c7edb17790c01dd5c627b02bb92bbeeda364"),
    "sv-SE": ("cmu5w9ub700b4o10780ew57mk", "5ea3bb11f4100b05c90f11d6b6b9efd4e35124056eeb3bcc5a1171f448323dd0"),
    "ky": ("cmu5w5qzx00c4nq07qmpfv22g", "ef106199b9d2b25d3efc5eab881e95b36a71cff3b1873af4d2e787e4934d65c0"),
    "bn": ("cmu5kut0d00pomh07yay2f34r", "3cdebe9005bef6b53d829e0f544297c04df0b1b801c170b1be23adfc917783a6"),
    "tt": ("cmu62fysg00noo1077ljif8fv", "9359ee54c4b865045a04f46379cf36baeb472429af6f7ba1ac85387ba1a0b8f1"),
    "cv": ("cmu5vpqnt00a5o107k6xz3bk2", "351f51f9d3f99db1d2c775102332b5c03091e80dd0b733945e9e4d9119c23003"),
    "hy-AM": ("cmu5wakny00b8o107u9ju02nk", "2d9573674ef33eb9b567aa0026eb44a73b2dbff77b9232836afe140f2c8f43b4"),
    "ro": ("cmu5w65wf00ccnq07n7jxral6", "357d75108d8138bd0161f3dacec8b38b755b3808a32a0611ff0c5c5bdb48977d"),
    "el": ("cmu62fru800nko107kvyi4hrc", "fd65eeeb709956a892c19517e6a6582692a429f220a486d35957abf8cf2a4f49"),
    "fi": ("cmu5vgfuk00aunq07xzlhzvxg", "e8085e8b3c694548201f60091cd92d4dcf98b982f4ae5495ba23e4d9ee09a8b8"),
    "bg": ("cmu5v8f750091o1076gkcd8xg", "4f519306865b271c77ddd7a2aaf06f936fdceec47a80717669b993815156cef9"),
    "sr": ("cmu5thgz30067nq07e6rsuwsx", "a2e92c7b29302fac316e886ec3ff05dd26d9c090ed7e7217106be5633e8f2fba"),
    "gn": ("cmu62cmtm00ngo107bh2jh3ze", "d6b96255ae1500cd9545a35aa27242e1b3030726e9dd838fbe3234856fb72196"),
    "hsb": ("cmu5uo2vm008nnq07xzwow03j", "594541b8de26332cbfaac75f14d6ca6596cc2a3c82352151055ef85583909c8c"),
    "vi": ("cmu5vci3w00agnq07hy4nbauv", "dec3d030eaa3dad9eba59336da72a9e6b4c3aa8119b3ba53ab72a59db6c6b88c"),
    "ha": ("cmu5u0ood006tnq07tmgdpgwi", "22415eda54a77a4dcd472d3f018da03772799ef1150c8e7405384df07ff6dfbc"),
    "ko": ("cmu5rwtaf003dnq07e7n3q6tf", "28fa978a9d48ba63b311760bb3eaf2db663c42da2287951f935bd92534990d07"),
    "kk": ("cmu5pcyw70104mi07526gdxh5", "954fcd07156d260f2fa4ebf50282b55aa78587b523994b48fc843546c950011f"),
}
CV_SPLITS = ("train", "dev", "test", "validated")
# Owner, 2026-11-03: at most 30 h per speaker and dataset, applied by the dataset generator, not here.
CV_SPEAKER_CAP_S = 30 * 3600
# The API sits behind Cloudflare, which rejects urllib's default User-Agent (error 1010).
USER_AGENT = "openvad-fetch"


def mdc_api_key() -> str:
    key = os.environ.get("MDC_API_KEY", "")
    if not key:
        raise RuntimeError(
            "Common Voice needs a Mozilla Data Collective API key: create one at "
            "https://mozilladatacollective.com/api-reference and set MDC_API_KEY (just reads it from .env)"
        )
    return key


def seconds_until_reset(now: dt.datetime) -> float:
    """MDC allows 7 dataset downloads per organisation and day and resets at midnight UTC; wait 10 minutes past it."""
    midnight = (now + dt.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return (midnight - now).total_seconds() + 600


def download_session(dataset_id: str, key: str) -> dict:
    """Ask the API for a short-lived download URL; over the daily limit (429) wait for the reset and ask again."""
    req = urllib.request.Request(
        f"{CV_API}/datasets/{dataset_id}/download",
        method="POST",
        headers={"Authorization": f"Bearer {key}", "User-Agent": USER_AGENT},
    )
    while True:
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 403:
                raise PermissionError(
                    f"MDC refused {dataset_id}: accept its terms at https://mozilladatacollective.com/datasets/{dataset_id}"
                ) from e
            if e.code != 429:
                raise
            wait = seconds_until_reset(dt.datetime.now(dt.UTC))
            print(f"MDC daily download limit reached; waiting {wait / 3600:.1f} h for the reset at midnight UTC")
            time.sleep(wait)


def saved_session(path: Path) -> dict | None:
    """The saved session unless it expires within 5 minutes; without `expiresAt` it counts as valid."""
    if not path.exists():
        return None
    try:
        session = json.loads(path.read_text())
    except json.JSONDecodeError:
        session = None
    if not isinstance(session, dict) or "downloadUrl" not in session:
        path.unlink()  # a crash mid-write leaves a truncated file
        return None
    if expires := session.get("expiresAt"):
        expires_at = dt.datetime.fromisoformat(expires)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=dt.UTC)
        if expires_at - dt.datetime.now(dt.UTC) < dt.timedelta(minutes=5):
            return None
    return session


def download_archive(dataset_id: str, key: str, sha256: str, archive: Path, session_file: Path) -> None:
    """Fetch one archive, reusing the saved download link (every new link counts against the daily limit)."""
    headers = {"User-Agent": USER_AGENT}
    if (session := saved_session(session_file)) is not None:
        try:
            download_ranges(session["downloadUrl"], archive, sha256=sha256, headers=headers)
            return
        except urllib.error.HTTPError as e:
            if e.code != 403:
                raise
            session_file.unlink()  # expired or revoked
    for attempt in (1, 2):
        session = download_session(dataset_id, key)
        if session.get("checksum") not in (None, sha256):
            raise RuntimeError(f"Common Voice {dataset_id}: MDC now serves {session['checksum']}, not {CV_RELEASE}")
        # The link is a credential: create the file private. Mode 0o600 may not hold on an SMB share whose mount
        # options fix file modes.
        session_file.unlink(missing_ok=True)
        with os.fdopen(os.open(session_file, os.O_WRONLY | os.O_CREAT, 0o600), "w") as f:
            json.dump(session, f)
        try:
            download_ranges(session["downloadUrl"], archive, sha256=sha256, headers=headers)
            return
        except urllib.error.HTTPError as e:
            if e.code != 403:
                raise
            session_file.unlink()
            if attempt == 2:  # a second fresh link refused: not an expired link
                raise


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def split_stats(locale_dir: Path) -> dict[str, dict]:
    """Clips and speakers per official split, and the train pool: every validated clip of a speaker in neither dev
    nor test (ADR 0023), with its hours, the ten largest speakers and the hours left under the per-speaker cap."""
    rows = {split: read_tsv(locale_dir / f"{split}.tsv") for split in CV_SPLITS}
    stats: dict[str, dict] = {
        s: {"clips": len(r), "speakers": len({x["client_id"] for x in r})} for s, r in rows.items()
    }
    held_out = {x["client_id"] for split in ("dev", "test") for x in rows[split]}
    # Every release since 13.0 ships clip_durations.tsv; without it the pool hours stay 0 rather than fail a long fetch.
    dur_file = locale_dir / "clip_durations.tsv"
    durations = {r["clip"]: int(r["duration[ms]"]) / 1000 for r in read_tsv(dur_file)} if dur_file.exists() else {}
    per_speaker: collections.defaultdict[str, float] = collections.defaultdict(float)
    for x in rows["validated"]:
        if x["client_id"] not in held_out:
            per_speaker[x["client_id"]] += durations.get(x["path"], 0.0)
    stats["train_pool"] = {
        "speakers": len(per_speaker),
        "hours": round(sum(per_speaker.values()) / 3600, 1),
        "top_speaker_hours": [round(s / 3600, 1) for s in sorted(per_speaker.values(), reverse=True)[:10]],
        "hours_under_cap": round(sum(min(s, CV_SPEAKER_CAP_S) for s in per_speaker.values()) / 3600, 1),
    }
    return stats


def extract_validated(archive: Path, out: Path) -> tuple[Path, int]:
    """Extract the metadata and only the validated clips (invalidated and unvalidated clips are not speech labels).

    Two passes, because the TSVs may come after the clips in the archive; each streams the archive anew, as members
    of a stream cannot be revisited. A clip already on disk at full size is skipped, so a rerun after a kill resumes.
    Returns the locale directory and the number of validated clips missing from the archive.
    """
    with tarfile.open(archive, "r|gz") as tar:
        for m in tar:
            if "/clips/" not in m.name:
                tar.extract(m, out, filter=tar_filter)
    locale_dirs = [p.parent for p in out.glob("**/validated.tsv")]
    if len(locale_dirs) != 1:
        raise RuntimeError(f"Common Voice: expected one validated.tsv under {out}, found {len(locale_dirs)}")
    keep = {row["path"] for row in read_tsv(locale_dirs[0] / "validated.tsv")}
    present = 0
    with tarfile.open(archive, "r|gz") as tar:
        for m in tar:
            if "/clips/" in m.name and Path(m.name).name in keep:
                dest = out / m.name
                if not (dest.is_file() and dest.stat().st_size == m.size):
                    tar.extract(m, out, filter=tar_filter)
                present += 1
    if not present:
        raise RuntimeError(f"Common Voice: none of the {len(keep)} validated clips is in {archive.name}, kept")
    missing = len(keep) - present
    if missing:
        print(f"Common Voice: {missing} of {len(keep)} validated clips are not in {archive.name}")
    return locale_dirs[0], missing


def fetch_commonvoice() -> None:
    out = raw_dir("common-voice")
    key = mdc_api_key()
    locales = {}
    for locale, (dataset_id, sha256) in CV_LOCALES.items():
        target = out / locale
        marker = out / f".{locale}-extracted"
        archive = cache_dir() / f"common-voice-{CV_RELEASE}-{locale}.tar.gz"
        session_file = cache_dir() / f"common-voice-{CV_RELEASE}-{locale}.session.json"
        # The marker holds the missing-clip count; one written by older code is empty and the count unknown.
        if marker.exists():
            missing = int(marker.read_text() or -1)
        else:
            if archive.exists():
                try:
                    _check(archive, None, sha256)
                except ValueError as e:
                    print(f"Common Voice {locale}: cached archive is damaged ({e}); downloading it again")
                    archive.unlink()
            if not archive.exists():
                download_archive(dataset_id, key, sha256, archive, session_file)
            _, missing = extract_validated(archive, target)
            marker.write_text(str(missing))
        # 53 archives are 670 GB, so each goes as soon as its clips are out (also after a crash past the marker).
        archive.unlink(missing_ok=True)
        session_file.unlink(missing_ok=True)
        (locale_dir,) = (p.parent for p in target.glob("**/validated.tsv"))
        locales[locale] = {
            "mdc_dataset": dataset_id,
            "sha256": sha256,
            "dir": locale_dir.relative_to(out).as_posix(),
            "missing_clips": None if missing < 0 else missing,
            "splits": split_stats(locale_dir),
        }
    info = {
        "dataset": "common-voice",
        "release": CV_RELEASE,
        "licence": "CC0-1.0; MDC terms: no re-hosting or re-sharing, no attempt to identify speakers",
        "use": "train (augmented and telephone channel), dev, test (ADR 0023): official dev and test; train from every "
        "validated clip of the other speakers",
        "speaker_cap": {"hours_per_speaker": CV_SPEAKER_CAP_S / 3600, "applied_by": "the dataset generator"},
        "extracted": "metadata and validated clips only",
        "sources": [
            {"url": f"https://mozilladatacollective.com/datasets/{dataset_id}", "sha256": sha256}
            for dataset_id, sha256 in CV_LOCALES.values()
        ],
        "locales": locales,
    }
    write_manifest(out, info)
    for locale in CV_LOCALES:
        (out / f".{locale}-extracted").unlink(missing_ok=True)
