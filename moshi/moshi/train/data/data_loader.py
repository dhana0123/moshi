# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterator

from .dataset import build_dataset
from .interleaver import Batch, InterleavedTokenizer


def build_data_loader(
    instruct_tokenizer: InterleavedTokenizer,
    train_data: str,
    batch_size: int,
    seed: int | None,
    rank: int,
    world_size: int,
    is_eval: bool,
    shuffle: bool,
) -> Iterator[Batch]:
    dataset = build_dataset(
        pretrain_data=train_data,
        instruct_tokenizer=instruct_tokenizer,
        seed=seed,
        rank=rank,
        world_size=world_size,
        is_eval=is_eval,
        shuffle_pretrain=shuffle,
    )
    sample_list = []
    for sample in dataset:
        sample_list.append(sample)
        if len(sample_list) == batch_size:
            yield Batch.collate(sample_list)
            sample_list = []
