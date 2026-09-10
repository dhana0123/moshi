# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Model loading utilities for server inference."""

import traceback
import asyncio
import httpx
import logging
import os
import random
import sys

import torch
import numpy as np
from safetensors.torch import load

from ..models import loaders, LMGen, LMModel
from ..conditioners import ConditionTensors, ConditionAttributes

logger = logging.getLogger(__name__)


def get_condition_tensors(
    model_type: str,
    lm: LMModel,
    batch_size: int,
    cfg_coef: float,
    reference_text: str | None = None,
    first_speaker: str = "model",
) -> ConditionTensors:
    condition_tensors = {}
    if reference_text is None:
        reference_text = ""
    if lm.condition_provider is not None and lm.condition_provider.conditioners:
        conditions: list[ConditionAttributes] | None = None
        if model_type == "hibiki":
            conditions = [ConditionAttributes(text={"description": "very_good"}, tensor={}) for _ in range(batch_size)]
            if cfg_coef != 1.0:
                # Extending the conditions with the negatives for the CFG.
                conditions += [
                    ConditionAttributes(text={"description": "very_bad"}, tensor={}) for _ in range(batch_size)
                ]
        elif model_type == "moshi":
            text_contributs = {}
            if reference_text is not None:
                # When reference text is "", this allows to apply the learnt_padding.
                # Otherwise, the text condition will be applied.
                text_contributs["reference_with_time"] = reference_text
            if first_speaker == "model":
                text_contributs["first_speaker"] = "SPEAKER_MAIN"
            elif first_speaker == "user":
                text_contributs["first_speaker"] = "SPEAKER_OTHER"
            else:
                raise ValueError(f"Invalid first speaker: {first_speaker}")
            conditions = [
                ConditionAttributes(
                    text=text_contributs,
                    tensor={},
                )
                for _ in range(batch_size)
            ]
        else:
            raise RuntimeError(f"Model expects conditioning but model type {model_type} is not supported.")
        assert conditions is not None
        prepared = lm.condition_provider.prepare(conditions)
        condition_tensors = lm.condition_provider(prepared)
    return condition_tensors


async def get_conditioning_remote_async(
    text: str,
    encoder_url: str = "http://localhost:8001",
    timeout: float = 30.0,
) -> torch.Tensor:
    """
    Get condition tensors from a remote Encoder service (async).
    """
    if text is None:
        text = ""

    logger.info(f"[Remote Encoder] Sending request to {encoder_url}/embed")
    logger.info(f"[Remote Encoder] text='{text[:80]}'")

    start_time = asyncio.get_event_loop().time()

    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.post(
                f"{encoder_url}/embed",
                json={"text": text},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            logger.error(f"[Remote Encoder] HTTP error: {e.response.status_code} - {e.response.text}")
            logger.error(traceback.format_exc())
            raise
        except httpx.RequestError as e:
            logger.error(f"[Remote Encoder] Request error: {e}")
            logger.error(traceback.format_exc())
            raise

    elapsed = asyncio.get_event_loop().time() - start_time
    logger.info(f"[Remote Encoder] Received response in {elapsed:.3f}s")

    tensor = load(response.content)
    if "tensor" not in tensor:
        raise ValueError("Response safetensors missing 'tensor' entry")
    condition_tensor = tensor["tensor"]
    logger.info(f"Deserialized condition tensor: {condition_tensor.shape} {condition_tensor.dtype}")
    return condition_tensor


def load_models(args):
    device = getattr(args, "device", None)
    if device is None:
        device = "cuda:0"
    cfg_coef = getattr(args, "cfg_coef", 1.0)
    batch_size = getattr(args, "batch_size", 1)

    logger.info("retrieving moshi checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        args.hf_repo, args.moshi_weight, args.mimi_weight, args.tokenizer, lora_weights=None, config_path=args.config
    )

    reference_encoder_url = os.environ.get("REFERENCE_ENCODER_URL")
    if reference_encoder_url:
        skip_conditioners = ["reference_with_time"]
    else:
        skip_conditioners = []

    text_tokenizer = checkpoint_info.get_text_tokenizer()

    logger.info(f"loading mimi on {device}")
    mimi = checkpoint_info.get_mimi(device=device)
    mimi.set_profile(False)
    logger.info("mimi loaded")

    logger.info(f"loading moshi on {device}")
    lm = checkpoint_info.get_moshi(device=device, dtype=args.dtype, fuse_lora=True, skip_conditioners=skip_conditioners)
    logger.info("moshi loaded")

    logger.info("constructing moshi lm gen")
    condition_tensors = get_condition_tensors(
        model_type="moshi",
        lm=lm,
        batch_size=batch_size,
        cfg_coef=1.0,
        reference_text=None,
        first_speaker=args.init_active_speaker,
    )
    lm_gen = LMGen(
        lm,
        cfg_coef=cfg_coef,
        condition_tensors=condition_tensors,
        force_streaming_sum=True,
        **checkpoint_info.lm_gen_config,
    )
    logger.info("moshi lm gen constructed")

    return mimi, text_tokenizer, lm_gen


def setup_logging(level: int, formatter: logging.Formatter | None = None) -> None:
    """Configure root logging. Pass a custom ``formatter`` (e.g. server ``_ColorFormatter``) for colored output."""
    handler = logging.StreamHandler()
    handler.setFormatter(
        formatter or logging.Formatter(fmt="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    )
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def get_reference_encoder_url() -> str:
    """Get and validate REFERENCE_ENCODER_URL."""
    reference_encoder_url = os.environ.get("REFERENCE_ENCODER_URL")
    if reference_encoder_url is None:
        logger.error("REFERENCE_ENCODER_URL environment variable must be set")
        logger.error("Please set it to the URL of the reference encoder service, e.g.:")
        logger.error("  export REFERENCE_ENCODER_URL=http://localhost:8001")
        sys.exit(1)
    return reference_encoder_url
