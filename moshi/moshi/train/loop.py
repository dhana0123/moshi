# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from torch.optim import AdamW, lr_scheduler

from ..architecture import MoshiSystem
from ..models.loaders import DEFAULT_REPO
from .args import TrainArgs
from .checkpointing import Checkpointer
from .data.data_loader import build_data_loader
from .data.interleaver import InterleavedTokenizer, Interleaver
from .distributed import BACKEND, avg_aggregate, get_rank, get_world_size, is_torchrun, set_device
from .eval import evaluate
from .loss import moshi_loss
from .mixed_precision import downcast_mixed_precision, prepare_mixed_precision, upcast_mixed_precision
from .utils import TrainState, set_random_seed
from .wrapped_model import build_param_groups, maybe_fsdp

logger = logging.getLogger("moshi.train")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def train(config: str) -> None:
    args = TrainArgs.load(config)
    if not args.train_data:
        raise SystemExit("Set train_data in the YAML to a jsonl (or directory of jsonl files).")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    set_random_seed(args.seed)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if is_torchrun():
        set_device()
        dist.init_process_group(backend=BACKEND)

    run_dir = Path(args.run_dir)
    if get_rank() == 0:
        if run_dir.exists() and not args.overwrite_run_dir:
            raise RuntimeError(f"Run dir {run_dir} exists. Set overwrite_run_dir: true or pick another path.")
        if run_dir.exists() and args.overwrite_run_dir:
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    param_dtype = getattr(torch, args.param_dtype)
    load_weight = args.init != "random"

    freeze = args.freeze or {"mimi": True}
    main_logger_info(f"Building MoshiSystem init={args.init} freeze={freeze}")
    system = MoshiSystem.from_pretrained(
        hf_repo=args.hf_repo or DEFAULT_REPO,
        device=device,
        dtype=param_dtype,
        load_weight=load_weight,
        freeze=freeze,
        gradient_checkpointing=args.gradient_checkpointing,
        moshi_weights=args.moshi_path,
        mimi_weights=args.mimi_path,
        tokenizer=args.tokenizer_path,
        config_path=args.config_path,
    )
    if system.text_tokenizer is None:
        raise RuntimeError("Text tokenizer missing; CheckpointInfo.get_text_tokenizer failed.")
    if not load_weight:
        system.lm.train()
        system.freeze_components(freeze)

    lm = maybe_fsdp(system.lm, param_dtype)
    system.lm = lm  # type: ignore[assignment]

    interleaver = Interleaver(
        system.text_tokenizer,
        system.frame_rate,
        lm.text_padding_token_id,
        lm.end_of_text_padding_id,
        lm.zero_token_id,
        keep_main_only=args.keep_main_only,
        audio_delay=args.audio_delay,
        proba=args.text_mask_proba,
        device=device,
    )
    interleaved = InterleavedTokenizer(system.mimi, interleaver, duration_sec=args.duration_sec)
    data_loader = build_data_loader(
        interleaved,
        train_data=args.train_data,
        batch_size=args.batch_size,
        seed=args.seed,
        rank=get_rank(),
        world_size=get_world_size(),
        is_eval=False,
        shuffle=args.shuffle,
    )
    eval_loader = None
    if args.do_eval and args.eval_data:
        eval_loader = build_data_loader(
            interleaved,
            train_data=args.eval_data,
            batch_size=args.batch_size,
            seed=None,
            rank=get_rank(),
            world_size=get_world_size(),
            is_eval=True,
            shuffle=False,
        )

    groups = build_param_groups(lm, system.mimi, args.optim)
    optimizer = AdamW(groups, betas=args.optim.betas, eps=1e-8, weight_decay=args.optim.weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in groups],
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
    )
    state = TrainState(args.max_steps)
    checkpointer = Checkpointer(
        model=lm,
        state=state,
        run_dir=run_dir,
        config=dataclasses.asdict(args),
        optimizer=optimizer,
        num_ckpt_keep=args.num_ckpt_keep,
    )
    prepare_mixed_precision(lm.parameters(), param_dtype=param_dtype, optim_dtype=torch.float32)
    lm.train()

    while state.step < args.max_steps:
        state.start_step()
        is_last = state.step == args.max_steps
        optimizer.zero_grad()
        loss = torch.tensor(0.0, device=device)
        n_tokens = 0
        for i in range(args.num_microbatches):
            batch = next(data_loader)
            codes = batch.codes.to(device)
            condition_tensors = None
            if batch.condition_attributes is not None and getattr(lm, "condition_provider", None) is not None:
                condition_tensors = lm.condition_provider.prepare(batch.condition_attributes)
            output = lm(codes=codes, condition_tensors=condition_tensors)
            mb_loss, text_loss, audio_loss = moshi_loss(
                output.text_logits,
                codes[:, : lm.audio_offset],
                output.text_mask,
                output.logits,
                codes[:, lm.audio_offset : lm.audio_offset + lm.dep_q],
                output.mask,
                text_padding_ids={lm.text_padding_token_id, lm.end_of_text_padding_id},
                first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
                text_padding_weight=args.text_padding_weight,
            )
            mb_loss.backward()
            loss = loss + mb_loss.detach()
            n_tokens += int(output.text_mask.numel() + output.mask.numel())
            if i < args.num_microbatches - 1 and torch.cuda.is_available():
                torch.cuda.synchronize()
        if args.num_microbatches > 1:
            loss = loss / args.num_microbatches
            for p in lm.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.div_(args.num_microbatches)
        upcast_mixed_precision(lm.parameters(), optim_dtype=torch.float32)
        torch.nn.utils.clip_grad_norm_(lm.parameters(), args.max_norm)
        optimizer.step()
        downcast_mixed_precision(lm.parameters(), param_dtype=param_dtype)
        scheduler.step()
        avg_loss = avg_aggregate(loss.item()) if torch.cuda.is_available() and dist.is_initialized() else loss.item()
        state.end_step(n_tokens)
        if state.step % args.log_freq == 0 and get_rank() == 0:
            logger.info(
                "step %s/%s loss=%.4f text=%.4f audio=%.4f lr=%s",
                state.step,
                args.max_steps,
                avg_loss,
                float(text_loss.detach()),
                float(audio_loss.detach()),
                scheduler.get_last_lr(),
            )
        if args.do_eval and eval_loader is not None and args.eval_freq > 0 and (
            state.step % args.eval_freq == 0 or is_last
        ):
            evaluate(lm, eval_loader, state, args, lm)
        if args.do_ckpt and (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0 or is_last):
            checkpointer.save_checkpoint(dtype=param_dtype)

    main_logger_info("done")


def packaged_config() -> Path:
    return Path(__file__).resolve().parent / "configs" / "default_7b.yaml"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train Moshi (default architecture scaffold).")
    parser.add_argument(
        "--config",
        default=str(packaged_config()),
        help="YAML config path (default: packaged default_7b.yaml)",
    )
    ns = parser.parse_args(argv)
    train(ns.config)


if __name__ == "__main__":
    main()
