# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Upload packed Inner Monologue data to a **private** Hugging Face dataset."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("moshi.prepare.push")

DEFAULT_DATASET_NAME = "moshi-indic-inner-monologue"

DATASET_CARD = """---
license: other
private: true
task_categories:
  - audio-to-audio
tags:
  - moshi
  - inner-monologue
  - duplex
  - indic
---

# Moshi Inner Monologue train set (private)

Stereo 24 kHz clips (left = agent, right = user) plus agent-only word alignments
for Moshi Inner Monologue training. **No dialogue control tokens.**

## Layout

- `train.jsonl` — `{"path": "wav/....wav", "duration": ...}` (paths relative after upload)
- `wav/*.wav` + sibling `wav/*.json` alignments
- `dataset_meta.json`

Source audio may include IndicVoices conversational rows; respect upstream licenses
and keep this repo **private**.
"""


def resolve_dataset_repo_id(hf_dataset: str | None) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("pip install huggingface_hub to use --push-to-hub") from exc

    api = HfApi()
    try:
        info = api.whoami()
    except Exception as exc:
        raise SystemExit(
            "Hugging Face auth failed. Set HF_TOKEN or run `huggingface-cli login`."
        ) from exc
    username = info.get("name") if isinstance(info, dict) else None
    if not username:
        raise SystemExit("Could not resolve HF username from whoami().")
    if hf_dataset:
        if "/" in hf_dataset:
            return hf_dataset
        return f"{username}/{hf_dataset}"
    return f"{username}/{DEFAULT_DATASET_NAME}"


def write_dataset_readme(out_dir: Path) -> Path:
    path = out_dir / "README.md"
    if not path.exists():
        path.write_text(DATASET_CARD, encoding="utf-8")
    return path


def rewrite_jsonl_relative_paths(out_dir: Path) -> None:
    """Make train.jsonl paths relative to out_dir for Hub consumers."""
    jsonl = out_dir / "train.jsonl"
    if not jsonl.exists():
        return
    import json

    rows = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        p = Path(rec["path"])
        try:
            rel = p.resolve().relative_to(out_dir.resolve())
            rec["path"] = rel.as_posix()
        except ValueError:
            # Keep basename under wav/ if absolute path was outside out_dir
            rec["path"] = f"wav/{p.name}"
        rows.append(rec)
    with jsonl.open("w", encoding="utf-8") as f:
        for rec in rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def push_private_dataset(
    out_dir: Path,
    *,
    hf_dataset: str | None = None,
    include_tokens: bool = False,
) -> str:
    """Create/update a private dataset repo and upload ``out_dir``.

    Returns the repo_id.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("pip install huggingface_hub to use --push-to-hub") from exc

    out_dir = Path(out_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(out_dir)

    repo_id = resolve_dataset_repo_id(hf_dataset)
    write_dataset_readme(out_dir)
    rewrite_jsonl_relative_paths(out_dir)

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=True,
        exist_ok=True,
    )

    ignore = [".git*", "**/.DS_Store"]
    if not include_tokens:
        ignore.append("tokens/**")

    logger.info("Uploading %s → private dataset %s", out_dir, repo_id)
    api.upload_folder(
        folder_path=str(out_dir),
        repo_id=repo_id,
        repo_type="dataset",
        ignore_patterns=ignore,
    )
    url = f"https://huggingface.co/datasets/{repo_id}"
    logger.info("Uploaded private dataset: %s", url)
    print(url)
    return repo_id
