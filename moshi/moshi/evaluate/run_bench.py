# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Generate Moshi outputs for FullDuplexBench categories.

Walks ``--bench-root/<dataset>/*.wav``, runs streaming Moshi inference, and
writes ``--results-root/<dataset>/<id>.wav`` plus a sidecar JSON for scoring.

Stock Moshiko/Moshika have no text system-prompt conditioner, so PersonaPlex-style
prompts are not applied. Input is user audio only; output is Moshi audio +
inner-monologue text tokens.

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


def _reset_state(state: InferenceState) -> None:
    state.mimi.reset_streaming()
    state.lm_gen.reset_streaming()
    # LMGen.reset clears offsets but leaves the delay cache; wipe it so samples
    # do not leak tokens into the next file.
    gen_state = state.lm_gen._streaming_state
    if gen_state is not None and hasattr(gen_state, "cache"):
        gen_state.cache.fill_(state.lm_gen.lm_model.ungenerated_token_id)
        gen_state.offset_cpu = 0


def run_one(
    state: InferenceState,
    wav_path: Path,
    out_wav: Path,
    out_json: Path,
    input_path: Path | None,
) -> None:
    _reset_state(state)
    in_pcms, _ = sphn.read(str(wav_path), sample_rate=state.mimi.sample_rate)
    in_pcms = torch.from_numpy(in_pcms).to(device=state.device)
    # Mono, batch size 1.
    in_pcms = in_pcms[None, 0:1]

    out_items = state.run(in_pcms)
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
        log("info", f"{dataset}: {len(wavs)} wav files")
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
                    run_one(state, wav_path, out_wav, out_json, meta)
            except Exception as err:
                logger.exception("Failed on %s: %s", wav_path, err)

    log("info", f"Done. Results under {results_root}")


if __name__ == "__main__":
    main()
