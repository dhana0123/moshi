# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Generate Moshi outputs for FullDuplexBench categories.

Walks ``--bench-root/<dataset>/*.wav``, runs streaming Moshi inference with the
PersonaPlex Hybrid System Prompt (voice + text system prompt), and writes
``--results-root/<dataset>/<id>.wav`` plus a sidecar JSON for scoring.

Text prompts follow the PersonaPlex FullDuplexBench guidance:

- Pause / Backchannel / Turn taking: ``You enjoy having a good conversation.``
- User Interruption: wise-and-friendly-teacher assistant prompt

Expected bench layout (user-supplied FullDuplexBench dump, not in git)::

    bench-root/
      synthetic_pause_handling/   *.wav
      candor_pause_handling/      *.wav
      icc_backchannel/            *.wav
      candor_turn_taking/         *.wav + metadata *.json
      synthetic_user_interruption/ *.wav + metadata *.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import sphn
import torch

from ..client_utils import log
from ..models import loaders
from ..run_inference import InferenceState, seed_all

logger = logging.getLogger(__name__)

DATASET_NAMES = [
    "synthetic_pause_handling",
    "candor_pause_handling",
    "icc_backchannel",
    "candor_turn_taking",
    "synthetic_user_interruption",
]

# Datasets that need a sidecar metadata JSON next to (or matching) the wav.
METADATA_DATASETS = {
    "candor_turn_taking",
    "synthetic_user_interruption",
}

PROMPT_CONVERSATION = "You enjoy having a good conversation."
PROMPT_INTERRUPTION = (
    "You are a wise and friendly teacher. Answer questions or provide advice "
    "in a clear and engaging way."
)

DATASET_TEXT_PROMPT = {
    "synthetic_pause_handling": PROMPT_CONVERSATION,
    "candor_pause_handling": PROMPT_CONVERSATION,
    "icc_backchannel": PROMPT_CONVERSATION,
    "candor_turn_taking": PROMPT_CONVERSATION,
    "synthetic_user_interruption": PROMPT_INTERRUPTION,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bench-root",
        type=Path,
        required=True,
        help="Root directory containing FullDuplexBench dataset folders.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/fullduplex"),
        help="Where to write wav + json outputs (default: %(default)s).",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        default=None,
        help=f"Datasets to run (default: all of {DATASET_NAMES}).",
    )
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--moshi-weight", type=str, default=None)
    parser.add_argument("--mimi-weight", type=str, default=None)
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=loaders.DEFAULT_REPO,
        help="HF repo for Moshi (default: Moshiko).",
    )
    parser.add_argument("--config", type=str, default=None, help="Optional LM config JSON.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--half",
        action="store_const",
        const=torch.float16,
        default=torch.bfloat16,
        dest="dtype",
        help="Use float16 instead of bfloat16.",
    )
    parser.add_argument("--cfg-coef", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--text-prompt",
        type=str,
        default=None,
        help="Override dataset Hybrid System Prompt role text.",
    )
    parser.add_argument(
        "--voice-prompt",
        type=str,
        default="",
        help="Voice wav or .pt codes filename (joined with --voice-prompt-dir).",
    )
    parser.add_argument(
        "--voice-prompt-dir",
        type=str,
        default=None,
        help="Directory containing voice prompts.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples that already have a results json.",
    )
    return parser.parse_args()


def _find_metadata(wav_path: Path, dataset: str) -> Path | None:
    """Locate FullDuplexBench metadata JSON for turn-taking / interruption."""
    candidates = [
        wav_path.with_suffix(".json"),
        wav_path.parent / f"{wav_path.stem}_meta.json",
        wav_path.parent / "metadata" / f"{wav_path.stem}.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    # Some dumps keep a parallel meta folder with the same stem.
    parent_meta = wav_path.parent.parent / f"{dataset}_meta" / f"{wav_path.stem}.json"
    if parent_meta.exists():
        return parent_meta
    return None


def _text_tokens_to_pieces(
    text_tokenizer, text_tokens: torch.Tensor
) -> list[str]:
    pieces: list[str] = []
    for tid in text_tokens.tolist():
        if tid in (0, 3):
            continue
        piece = text_tokenizer.id_to_piece(int(tid))
        pieces.append(piece)
    return pieces


def run_one(
    state: InferenceState,
    wav_path: Path,
    out_wav: Path,
    out_json: Path,
    input_path: Path | None,
    text_prompt: str = "",
    voice_path: str | None = None,
) -> None:
    in_pcms, _ = sphn.read(str(wav_path), sample_rate=state.mimi.sample_rate)
    in_pcms = torch.from_numpy(in_pcms).to(device=state.device)
    # Mono, batch size 1.
    in_pcms = in_pcms[None, 0:1]

    out_items = state.run(in_pcms, text_prompt=text_prompt, voice_path=voice_path)
    if not out_items:
        raise RuntimeError(f"No output generated for {wav_path}")

    text_tokens, out_pcm = out_items[0]
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sphn.write_wav(str(out_wav), out_pcm[0].numpy(), sample_rate=state.mimi.sample_rate)

    model_text = _text_tokens_to_pieces(state.text_tokenizer, text_tokens)
    payload = {
        "input_path": str(input_path.resolve()) if input_path is not None else "",
        "model_text": model_text,
        "user_text": None,
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    seed_all(args.seed)

    bench_root = args.bench_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    datasets = args.dataset_names or DATASET_NAMES

    if not bench_root.exists():
        raise SystemExit(f"Bench root does not exist: {bench_root}")

    log("info", "retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo,
        args.moshi_weight,
        args.mimi_weight,
        args.tokenizer,
        args.config,
    )
    log("info", "loading mimi")
    mimi = checkpoint_info.get_mimi(device=args.device)
    text_tokenizer = checkpoint_info.get_text_tokenizer()
    log("info", "loading moshi")
    lm = checkpoint_info.get_moshi(device=args.device, dtype=args.dtype)

    state = InferenceState(
        checkpoint_info,
        mimi,
        text_tokenizer,
        lm,
        batch_size=1,
        cfg_coef=args.cfg_coef,
        device=args.device,
        **checkpoint_info.lm_gen_config,
    )

    voice_path = args.voice_prompt or None
    if voice_path and args.voice_prompt_dir:
        voice_path = str(Path(args.voice_prompt_dir) / args.voice_prompt)
        if not Path(voice_path).exists():
            raise FileNotFoundError(f"Voice prompt not found: {voice_path}")

    for dataset in datasets:
        if dataset not in DATASET_NAMES:
            log("warning", f"Unknown dataset {dataset}, skipping")
            continue
        src_dir = bench_root / dataset
        if not src_dir.is_dir():
            log("warning", f"Missing dataset folder {src_dir}, skipping")
            continue
        dst_dir = results_root / dataset
        dst_dir.mkdir(parents=True, exist_ok=True)

        wavs = sorted(src_dir.rglob("*.wav"))
        text_prompt = args.text_prompt or DATASET_TEXT_PROMPT.get(dataset, PROMPT_CONVERSATION)
        log("info", f"{dataset}: {len(wavs)} wav files; prompt={text_prompt!r}")
        for wav_path in wavs:
            rel = wav_path.relative_to(src_dir)
            out_stem = dst_dir / rel.with_suffix("")
            out_wav = Path(str(out_stem) + ".wav")
            out_json = Path(str(out_stem) + ".json")
            out_wav.parent.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and out_json.exists() and out_wav.exists():
                continue

            meta = None
            if dataset in METADATA_DATASETS:
                meta = _find_metadata(wav_path, dataset)
                if meta is None:
                    log(
                        "warning",
                        f"No metadata JSON for {wav_path}; turn-taking/interrupt "
                        "scoring will fail unless input_path is set later.",
                    )

            try:
                log("info", f"Running {wav_path.name}")
                with torch.no_grad():
                    run_one(
                        state,
                        wav_path,
                        out_wav,
                        out_json,
                        meta,
                        text_prompt=text_prompt,
                        voice_path=voice_path,
                    )
            except Exception as err:
                logger.exception("Failed on %s: %s", wav_path, err)

    log("info", f"Done. Results under {results_root}")


if __name__ == "__main__":
    main()
