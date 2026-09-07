# MahaDhwani local sample audio

Tiny listen/smoke helper for [AI4Bharat/MahaDhwani](https://github.com/AI4Bharat/MahaDhwani).

MahaDhwani is **audio** (YouTube → `yt-dlp --extract-audio` → MP3), not video files.
Their cloud layout is `mahadhwani/<Lang>/mp3/*.mp3`. This script downloads the same
way for a few short clips so you can hear them locally.

## Default set

- **4 languages:** Hindi, Tamil, Telugu, Kannada  
- **2 samples each** (short clips ~20–180s when metadata allows)

## Setup

```bash
pip install yt-dlp
# ffmpeg must be on PATH (for audio extract)
```

## Run

From repo root or this folder:

```bash
python data/mahadhwani/download_samples.py
```

Options:

```bash
# preview IDs only
python data/mahadhwani/download_samples.py --dry-run

# other languages / counts
python data/mahadhwani/download_samples.py --languages Hindi Marathi Bengali Tamil --samples-per-lang 2
```

Outputs:

```
data/mahadhwani/
  Hindi/*.mp3
  Tamil/*.mp3
  Telugu/*.mp3
  Kannada/*.mp3
  samples_manifest.json
```

The script checks (via `ffprobe` when available) that files are **audio-only** (no video stream).

Large media is gitignored; only this README + script are meant to be committed.
