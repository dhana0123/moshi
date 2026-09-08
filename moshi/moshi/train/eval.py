# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Iterator

import torch
import torch.distributed as dist

from .args import TrainArgs
from .data.interleaver import Batch
from .distributed import dist_ready, get_rank, get_world_size
from .loss import moshi_loss
from ..models.hybrid_prompt import apply_prefix_loss_mask
from .utils import TrainState

logger = logging.getLogger("moshi.train")


def evaluate(model, eval_data_loader: Iterator[Batch], state: TrainState, args: TrainArgs, lm):
    text_loss_sum = torch.tensor(0.0, device=next(lm.parameters()).device)
    audio_loss_sum = torch.tensor(0.0, device=text_loss_sum.device)
    n = torch.tensor(0, device=text_loss_sum.device)
    model.eval()
    for batch in eval_data_loader:
        n += 1
        if int(n.item()) > max(1, 40 // get_world_size()):
            break
        with torch.no_grad():
            output = lm(codes=batch.codes)
            text_mask, audio_mask = output.text_mask, output.mask
            if batch.prefix_frames is not None and int(batch.prefix_frames.max()) > 0:
                text_mask, audio_mask = apply_prefix_loss_mask(
                    text_mask, audio_mask, batch.prefix_frames.to(text_mask.device)
                )
            total, text_loss, audio_loss = moshi_loss(
                output.text_logits,
                batch.codes[:, : lm.audio_offset],
                text_mask,
                output.logits,
                batch.codes[:, lm.audio_offset : lm.audio_offset + lm.dep_q],
                audio_mask,
                text_padding_ids={lm.text_padding_token_id, lm.end_of_text_padding_id},
                first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
                text_padding_weight=args.text_padding_weight,
            )
            text_loss_sum += text_loss
            audio_loss_sum += audio_loss
    eval_loss = text_loss_sum + audio_loss_sum
    if dist_ready():
        dist.all_reduce(eval_loss)
        dist.all_reduce(text_loss_sum)
        dist.all_reduce(audio_loss_sum)
        dist.all_reduce(n)
    denom = max(int(n.item()), 1)
    state.this_eval_loss = (eval_loss / denom).item()
    state.this_eval_perplexity = (2 ** (eval_loss / denom)).item()
    state.this_text_loss = (text_loss_sum / denom).item()
    state.this_audio_loss = (audio_loss_sum / denom).item()
    if get_rank() == 0:
        logger.info(
            "eval loss=%.4f text=%.4f audio=%.4f ppl=%.2f",
            state.this_eval_loss,
            state.this_text_loss,
            state.this_audio_loss,
            state.this_eval_perplexity,
        )
    model.train()
