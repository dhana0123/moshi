# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import sphn

from .interleaver import InterleavedTokenizer, Sample


@dataclass
class DataDir:
    path: Path

    @property
    def jsonl_files(self):
        files = list(self.path.rglob("*jsonl"))
        if not files:
            raise FileNotFoundError(f"{self.path} has no .jsonl files")
        return files


@dataclass
class DataFile:
    path: Path

    @property
    def jsonl_files(self):
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        return [self.path]


def parse_data_sources(pretrain_data: str) -> tuple[list[DataDir | DataFile], list[float]]:
    seen: set[str] = set()
    sources: list[DataDir | DataFile] = []
    weights: list[float] = []
    for source in pretrain_data.strip().split(","):
        if not source:
            continue
        source_items = source.strip().split(":")
        if len(source_items) == 1:
            path_, weight = source_items[0], 1.0
        elif len(source_items) == 2:
            path_, weight = source_items[0], float(source_items[1])
        else:
            raise ValueError(f"Bad data source {source!r}")
        if path_ in seen:
            raise ValueError(f"Duplicate data source {path_}")
        if Path(path_).is_dir():
            data: DataDir | DataFile = DataDir(path=Path(path_))
        elif Path(path_).is_file():
            data = DataFile(path=Path(path_))
        else:
            raise FileNotFoundError(path_)
        sources.append(data)
        weights.append(weight)
        seen.add(path_)
    total = sum(weights)
    return sources, [w / total for w in weights]


def build_dataset(
    pretrain_data: str,
    instruct_tokenizer: InterleavedTokenizer,
    seed: int | None,
    rank: int,
    world_size: int,
    is_eval: bool,
    shuffle_pretrain: bool = False,
) -> Iterator[Sample]:
    sources, probabilities = parse_data_sources(pretrain_data)
    shuffle = not is_eval and shuffle_pretrain
    iterators = [
        get_dataset_iterator(
            source,
            instruct_tokenizer=instruct_tokenizer,
            rank=rank,
            world_size=world_size,
            is_finite=is_eval,
            seed=seed,
            shuffle_at_epoch=shuffle,
        )
        for source in sources
    ]
    if is_eval:
        return itertools.chain.from_iterable(iterators)
    rng = np.random.RandomState(seed=np.array((seed or 0, rank)))
    return interleave_iterators(iterators, probabilities, rng)


def get_dataset_iterator(
    source: DataDir | DataFile,
    instruct_tokenizer: InterleavedTokenizer,
    rank: int,
    world_size: int,
    is_finite: bool,
    seed: int | None,
    shuffle_at_epoch: bool,
) -> Iterator[Sample]:
    epoch = 1
    while True:
        for jsonl_file in source.jsonl_files:
            dataset = sphn.dataset_jsonl(
                str(jsonl_file),
                duration_sec=instruct_tokenizer.duration_sec,
                num_threads=4,
                sample_rate=instruct_tokenizer.mimi.sample_rate,
                pad_last_segment=True,
            )
            if shuffle_at_epoch:
                dataset = dataset.shuffle(
                    with_replacement=False, skip=rank, step_by=world_size, seed=seed
                )
                seed = (seed or 0) + 1
            else:
                dataset = dataset.seq(skip=rank, step_by=world_size)
            for sample in dataset:
                wav = sample["data"][..., : sample["unpadded_len"]]
                yield instruct_tokenizer(wav, sample["start_time_sec"], sample["path"])
        if is_finite:
            break
        print(f"Rank {rank} finished epoch {epoch}")
        epoch += 1


def interleave_iterators(iterators, probabilities, rng):
    while True:
        it_id = rng.choice(range(len(iterators)), p=probabilities)
        yield next(iterators[it_id])


def load_jsonl_paths(path: Path, world_size: int, rank: int) -> list[str]:
    lines = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if idx % world_size == rank:
                lines.append(json.loads(line))
    return lines
