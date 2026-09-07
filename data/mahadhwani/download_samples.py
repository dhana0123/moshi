#!/usr/bin/env python3
"""Download a few MahaDhwani sample *audio* files locally (not video).

MahaDhwani ([AI4Bharat/MahaDhwani](https://github.com/AI4Bharat/MahaDhwani)) is a
raw-audio corpus built from public YouTube content. Their pipeline uses yt-dlp
with ``--extract-audio`` to store MP3; this script does the same for a tiny
listen/smoke set.

Default: 4 languages x 2 short clips each -> ``data/mahadhwani/<Lang>/*.mp3``

Requires: ``yt-dlp`` on PATH (and ``ffmpeg`` for audio extract).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_RAW = "https://raw.githubusercontent.com/AI4Bharat/MahaDhwani/master"
LANG_DIR = f"{REPO_RAW}/dataflow_pipeline/languages"

# Sensible defaults for an Indic duplex smoke listen.
DEFAULT_LANGUAGES = ["Hindi", "Tamil", "Telugu", "Kannada"]

# Prefer short clips so downloads stay small and easy to audition.
MIN_DURATION_SEC = 20
MAX_DURATION_SEC = 180


def _http_get(url: str, timeout: float = 120, max_bytes: int | None = None) -> bytes:
    headers = {"User-Agent": "moshi-mahadhwani-samples/1.0"}
    if max_bytes is not None:
        headers["Range"] = f"bytes=0-{max_bytes - 1}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read(max_bytes if max_bytes is not None else -1)


def _stream_metadata_rows(lang: str, max_scan: int = 5000):
    """Yield (video_id, duration_sec, domain) from MahaDhwani metadata CSV.

    Only the first ~1 MiB of the CSV is fetched (files can be tens of MB).
    """
    url = f"{LANG_DIR}/{lang}/video_ids_metadata_{lang}.csv"
    try:
        raw = _http_get(url, max_bytes=1_000_000)
    except urllib.error.HTTPError as err:
        raise RuntimeError(f"Failed to fetch metadata for {lang}: {err}") from err

    text = raw.decode("utf-8", errors="replace")
    # Drop trailing partial line from Range truncates.
    if not text.endswith("\n"):
        text = text.rsplit("\n", 1)[0]
    reader = csv.DictReader(io.StringIO(text))
    for i, row in enumerate(reader):
        if i >= max_scan:
            break
        vid = (row.get("id") or "").strip()
        if not vid:
            continue
        try:
            duration = float(row.get("duration(sec)") or 0)
        except ValueError:
            continue
        domain = (row.get("domain") or "").strip()
        yield vid, duration, domain


def _fallback_ids(lang: str, n: int) -> list[str]:
    url = f"{LANG_DIR}/{lang}/vids_list_{lang}.txt"
    # First ~64 KiB is enough for dozens of IDs.
    raw = _http_get(url, max_bytes=65_536)
    text = raw.decode("utf-8", errors="replace")
    if not text.endswith("\n"):
        text = text.rsplit("\n", 1)[0]
    ids = [line.strip() for line in text.splitlines() if line.strip()]
    return ids[:n]


def pick_video_ids(
    lang: str,
    n: int,
    min_dur: float,
    max_dur: float,
    max_scan: int,
) -> list[dict]:
    """Pick up to n short clips for a language."""
    chosen: list[dict] = []
    seen: set[str] = set()
    try:
        for vid, duration, domain in _stream_metadata_rows(lang, max_scan=max_scan):
            if vid in seen:
                continue
            if not (min_dur <= duration <= max_dur):
                continue
            seen.add(vid)
            chosen.append(
                {
                    "id": vid,
                    "duration_sec": duration,
                    "domain": domain,
                    "url": f"https://youtu.be/{vid}",
                }
            )
            if len(chosen) >= n:
                break
    except Exception as err:
        print(f"[{lang}] metadata CSV failed ({err}); falling back to vids_list", file=sys.stderr)

    if len(chosen) < n:
        for vid in _fallback_ids(lang, n * 5):
            if vid in seen:
                continue
            seen.add(vid)
            chosen.append(
                {
                    "id": vid,
                    "duration_sec": None,
                    "domain": "",
                    "url": f"https://youtu.be/{vid}",
                }
            )
            if len(chosen) >= n:
                break
    return chosen[:n]


def ensure_yt_dlp() -> str:
    path = shutil.which("yt-dlp")
    if not path:
        raise SystemExit(
            "yt-dlp not found on PATH. Install with:\n"
            "  pip install yt-dlp\n"
            "Also install ffmpeg for --extract-audio."
        )
    if not shutil.which("ffmpeg"):
        print(
            "Warning: ffmpeg not found; yt-dlp audio extract may fail.",
            file=sys.stderr,
        )
    return path


def download_audio(yt_dlp: str, video_id: str, out_mp3: Path, max_duration: float | None) -> bool:
    """Download best audio only as mp3 (mono 16 kHz), matching MahaDhwani pipeline."""
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    if out_mp3.exists() and out_mp3.stat().st_size > 0:
        print(f"  skip (exists): {out_mp3.name}")
        return True

    output_template = str(out_mp3.with_suffix(""))  # yt-dlp adds .mp3
    cmd = [
        yt_dlp,
        "-f",
        "bestaudio/best",
        "--extract-audio",
        "--audio-format",
        "mp3",
        "--audio-quality",
        "0",
        "--no-playlist",
        "-o",
        output_template,
        "--ppa",
        "ffmpeg:-ac 1 -ar 16000",
        f"https://youtu.be/{video_id}",
    ]
    # Skip long videos when duration was unknown at pick time.
    if max_duration is not None:
        cmd.extend(["--match-filter", f"duration <= {int(max_duration)}"])

    print(f"  downloading audio-only: {video_id} -> {out_mp3.name}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as err:
        print(f"  FAILED {video_id}: {err.stderr[-500:] if err.stderr else err}", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT {video_id}", file=sys.stderr)
        return False

    # yt-dlp may write exact path or path.mp3 depending on version/template.
    if not out_mp3.exists():
        candidates = list(out_mp3.parent.glob(f"{video_id}*"))
        audio_cands = [
            p for p in candidates if p.suffix.lower() in {".mp3", ".m4a", ".webm", ".opus", ".wav"}
        ]
        if audio_cands:
            audio_cands[0].replace(out_mp3)
        else:
            print(f"  FAILED: no audio file written for {video_id}", file=sys.stderr)
            return False

    if out_mp3.suffix.lower() in {".mp4", ".mkv", ".webm"} and not _looks_like_audio_only(out_mp3):
        print(f"  REJECTED video-looking file: {out_mp3}", file=sys.stderr)
        out_mp3.unlink(missing_ok=True)
        return False
    return out_mp3.exists() and out_mp3.stat().st_size > 0


def _looks_like_audio_only(path: Path) -> bool:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return path.suffix.lower() == ".mp3"
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v",
        "-show_entries",
        "stream=codec_type",
        "-of",
        "csv=p=0",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=30)
    except Exception:
        return path.suffix.lower() == ".mp3"
    # Empty means no video stream -> audio-only OK.
    return out.stdout.strip() == ""


def verify_audio_file(path: Path) -> dict:
    info = {"path": str(path), "bytes": path.stat().st_size, "audio_only": None, "duration_sec": None}
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        info["audio_only"] = path.suffix.lower() == ".mp3"
        return info

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type",
        "-of",
        "json",
        str(path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        payload = json.loads(out.stdout)
        streams = payload.get("streams") or []
        types = {s.get("codec_type") for s in streams}
        info["audio_only"] = "audio" in types and "video" not in types
        dur = (payload.get("format") or {}).get("duration")
        if dur is not None:
            info["duration_sec"] = round(float(dur), 2)
    except Exception as err:
        info["error"] = str(err)
        info["audio_only"] = path.suffix.lower() == ".mp3"
    return info


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--languages",
        nargs="+",
        default=DEFAULT_LANGUAGES,
        help=f"MahaDhwani language folder names (default: {DEFAULT_LANGUAGES})",
    )
    p.add_argument("--samples-per-lang", type=int, default=2, help="Clips per language (default: 2)")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=here,
        help="Output root (default: this folder)",
    )
    p.add_argument("--min-duration", type=float, default=MIN_DURATION_SEC)
    p.add_argument("--max-duration", type=float, default=MAX_DURATION_SEC)
    p.add_argument(
        "--max-scan",
        type=int,
        default=8000,
        help="Max metadata CSV rows to scan per language when picking short clips",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print chosen YouTube IDs; do not download",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root: Path = args.out_dir
    out_root.mkdir(parents=True, exist_ok=True)

    yt_dlp = None if args.dry_run else ensure_yt_dlp()

    manifest: list[dict] = []
    for lang in args.languages:
        print(f"\n=== {lang} ===")
        picks = pick_video_ids(
            lang,
            n=max(args.samples_per_lang * 3, args.samples_per_lang),
            min_dur=args.min_duration,
            max_dur=args.max_duration,
            max_scan=args.max_scan,
        )
        if not picks:
            print(f"[{lang}] no video IDs found", file=sys.stderr)
            continue

        lang_dir = out_root / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        for meta in picks:
            if saved >= args.samples_per_lang:
                break
            vid = meta["id"]
            out_mp3 = lang_dir / f"{lang.lower()}_{saved + 1:02d}_{vid}.mp3"
            entry = {**meta, "language": lang, "local_path": str(out_mp3)}
            print(
                f"  try {vid} dur={meta.get('duration_sec')} domain={meta.get('domain')!r}"
            )

            if args.dry_run:
                entry["status"] = "dry_run"
                manifest.append(entry)
                saved += 1
                continue

            ok = download_audio(
                yt_dlp,
                vid,
                out_mp3,
                max_duration=args.max_duration if meta.get("duration_sec") is None else None,
            )
            if ok:
                verify = verify_audio_file(out_mp3)
                entry["status"] = "ok"
                entry["verify"] = verify
                if verify.get("audio_only") is False:
                    print(f"  WARNING: {out_mp3.name} may contain a video stream", file=sys.stderr)
                else:
                    print(
                        f"  ok audio-only={verify.get('audio_only')} "
                        f"duration={verify.get('duration_sec')}s "
                        f"size={verify.get('bytes')} bytes"
                    )
                saved += 1
            else:
                entry["status"] = "failed"
            manifest.append(entry)

        if saved < args.samples_per_lang:
            print(
                f"[{lang}] only got {saved}/{args.samples_per_lang} samples",
                file=sys.stderr,
            )
    manifest_path = out_root / "samples_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    ok_n = sum(1 for m in manifest if m.get("status") in {"ok", "dry_run"})
    failed_n = sum(1 for m in manifest if m.get("status") == "failed")
    print(f"\nDone. listed/ok={ok_n} failed={failed_n}. Manifest: {manifest_path}")
    print("Play the .mp3 files under each language folder to listen/check.")


if __name__ == "__main__":
    main()
