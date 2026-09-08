# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Inner Monologue: word timestamps at 12.5 Hz with PAD / EPAD (paper §3.4.4)."""

from __future__ import annotations

import json
import math
import os
from collections import deque
from dataclasses import dataclass
from functools import reduce

import numpy as np
import sentencepiece
import torch

from ...conditioners import ConditionAttributes

Alignment = tuple[str, tuple[float, float], str]
TokenizedAlignment = tuple[list[int], tuple[float, float], str]


from ...models.hybrid_prompt import (
    HybridPromptConfig,
    build_hybrid_system_prefix,
    encode_wav_agent_codes,
    wrap_with_system_tags,
)


@dataclass
class Sample:
    codes: torch.Tensor
    condition_attributes: ConditionAttributes | None = None
    prefix_frames: int = 0


@dataclass
class Batch:
    codes: torch.Tensor
    condition_attributes: list[ConditionAttributes] | None = None
    prefix_frames: torch.Tensor | None = None

    @classmethod
    def collate(cls, batch: list[Sample]) -> "Batch":
        codes = torch.cat([b.codes for b in batch])
        prefix = torch.tensor([b.prefix_frames for b in batch], dtype=torch.long)
        if batch[0].condition_attributes is None:
            return Batch(codes, prefix_frames=prefix)
        return Batch(codes, [b.condition_attributes for b in batch], prefix_frames=prefix)


def tokenize(
    tokenizer: sentencepiece.SentencePieceProcessor,
    text: str,
    bos: bool = True,
    alpha: float | None = None,
):
    nl_piece = tokenizer.encode("\n")[-1]
    if alpha is not None:
        tokens = tokenizer.encode(
            text.split("\n"), enable_sampling=True, alpha=alpha, nbest_size=-1
        )
    else:
        tokens = tokenizer.encode(text.split("\n"))
    tokens = reduce(lambda a, b: [*a, nl_piece, *b], tokens)
    if bos:
        tokens = [tokenizer.bos_id(), *tokens]
    return tokens


class Interleaver:
    def __init__(
        self,
        tokenizer: sentencepiece.SentencePieceProcessor,
        audio_frame_rate: float,
        text_padding: int,
        end_of_text_padding: int,
        zero_padding: int,
        in_word_padding: int | None = None,
        keep_main_only: bool = True,
        main_speaker_label: str = "SPEAKER_MAIN",
        use_bos_eos: bool = False,
        keep_and_shift: bool = False,
        audio_delay: float = 0.0,
        proba: float = 1.0,
        device: str | torch.device = "cpu",
    ):
        self.tokenizer = tokenizer
        self.audio_frame_rate = audio_frame_rate
        self.text_padding = text_padding
        self.end_of_text_padding = end_of_text_padding
        self.zero_padding = zero_padding
        self.in_word_padding = self.text_padding if in_word_padding is None else in_word_padding
        self.keep_main_only = keep_main_only
        self.main_speaker_label = main_speaker_label
        self.use_bos_eos = use_bos_eos
        self.keep_and_shift = keep_and_shift
        self.audio_delay = audio_delay
        self.proba = proba
        self.device = device

    def _tokenize(self, alignments: list[Alignment]) -> list[TokenizedAlignment]:
        out = []
        for word, ts, speaker in alignments:
            toks = tokenize(self.tokenizer, word.strip(), bos=False)
            out.append((toks, ts, speaker))
        return out

    def _keep_main_only(
        self, alignments: list[TokenizedAlignment], main_speaker: str
    ) -> list[TokenizedAlignment]:
        return [a for a in alignments if a[2] == main_speaker]

    def _keep_those_with_duration(
        self, alignments: list[TokenizedAlignment]
    ) -> list[TokenizedAlignment]:
        return [a for a in alignments if a[1][0] < a[1][1]]

    def _add_delay(
        self, alignments: list[TokenizedAlignment]
    ) -> list[TokenizedAlignment]:
        return [
            (a[0], (a[1][0] - self.audio_delay, a[1][1] - self.audio_delay), a[2])
            for a in alignments
            if a[1][1] > self.audio_delay
        ]

    def _insert_bos_eos(
        self, alignments: list[TokenizedAlignment], main_speaker: str
    ) -> list[TokenizedAlignment]:
        out: list[TokenizedAlignment] = []
        last_speaker = None
        for toks, ts, speaker in alignments:
            toks = list(toks)
            if speaker == last_speaker:
                pass
            elif speaker == main_speaker:
                toks.insert(0, self.tokenizer.bos_id())
            elif last_speaker == main_speaker:
                toks.insert(0, self.tokenizer.eos_id())
            last_speaker = speaker
            out.append((toks, ts, speaker))
        return out

    def build_token_stream(
        self,
        alignments: list[TokenizedAlignment] | None,
        segment_duration: float,
    ) -> torch.Tensor:
        T = math.ceil(segment_duration * self.audio_frame_rate)
        if alignments is None:
            text_tokens = [self.zero_padding] * T
        else:
            text_tokens = [self.text_padding] * T
            i = 0
            to_append_stack: deque = deque()
            last_word_end = -1
            for t in range(T):
                while (
                    i < len(alignments)
                    and alignments[i][1][0] * self.audio_frame_rate < t + 1
                ):
                    tokenized = alignments[i][0]
                    last_word_end = int(alignments[i][1][1] * self.audio_frame_rate)
                    if self.keep_and_shift:
                        to_append_stack.extend(tokenized)
                    else:
                        to_append_stack = deque(tokenized)
                    i += 1
                if to_append_stack:
                    if t > 0 and text_tokens[t - 1] in [
                        self.text_padding,
                        self.in_word_padding,
                    ]:
                        text_tokens[t - 1] = self.end_of_text_padding
                    text_tokens[t] = to_append_stack.popleft()
                elif t <= last_word_end:
                    text_tokens[t] = self.in_word_padding
        if self.audio_delay < 0:
            prefix_length = int(self.audio_frame_rate * -self.audio_delay)
            text_tokens[:prefix_length] = [self.zero_padding] * prefix_length
        return torch.tensor(text_tokens, device=self.device).view(1, 1, -1)

    def prepare_item(
        self,
        alignments: list[Alignment] | None,
        segment_duration: float,
        main_speaker: str | None = None,
    ) -> torch.Tensor:
        if alignments is None:
            tokenized = None
        else:
            tokenized = self._tokenize(sorted(alignments, key=lambda x: x[1][0]))
            if self.keep_main_only:
                main_speaker = main_speaker or self.main_speaker_label
                tokenized = self._keep_main_only(tokenized, main_speaker)
            elif self.use_bos_eos:
                main_speaker = main_speaker or self.main_speaker_label
                tokenized = self._insert_bos_eos(tokenized, main_speaker)
            tokenized = self._keep_those_with_duration(tokenized)
            if self.audio_delay != 0:
                tokenized = self._add_delay(tokenized)
        return self.build_token_stream(tokenized, segment_duration)


def _dicho(alignment, val, i=0, j=None):
    if j is None:
        j = len(alignment)
    if i == j:
        return i
    k = (i + j) // 2
    if alignment[k][1][0] < val:
        return _dicho(alignment, val, k + 1, j)
    return _dicho(alignment, val, i, k)


class InterleavedTokenizer:
    def __init__(
        self,
        mimi,
        interleaver: Interleaver,
        duration_sec: float,
        hybrid: HybridPromptConfig | None = None,
        n_q: int | None = None,
        dep_q: int | None = None,
    ):
        self.mimi = mimi
        self.interleaver = interleaver
        self.duration_sec = duration_sec
        self.num_audio_frames = math.ceil(duration_sec * mimi.frame_rate)
        self.hybrid = hybrid or HybridPromptConfig()
        self.n_q = n_q
        self.dep_q = dep_q
        self._voice_cache: dict[str, torch.Tensor] = {}
        self._rng = np.random.default_rng()

    def _voice_codes(self, dep_q: int) -> torch.Tensor | None:
        files = self.hybrid.list_voice_files()
        if not files:
            return None
        path = files[int(self._rng.integers(0, len(files)))]
        cached = self._voice_cache.get(str(path))
        if cached is not None:
            return cached
        if path.suffix.lower() == ".pt":
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            codes = payload["codes"] if isinstance(payload, dict) and "codes" in payload else payload
            if not torch.is_tensor(codes):
                return None
            if codes.dim() == 3:
                codes = codes[0]
            encoded = codes.to(dtype=torch.long)
        else:
            try:
                import sphn
            except ImportError:
                return None
            wav, _ = sphn.read(str(path), sample_rate=self.mimi.sample_rate)
            encoded = encode_wav_agent_codes(self.mimi, wav).cpu()
        if encoded.shape[0] > dep_q:
            encoded = encoded[:dep_q]
        self._voice_cache[str(path)] = encoded
        return encoded

    def __call__(self, wav: np.ndarray, start_sec: float, path: str) -> Sample:
        device = next(self.mimi.parameters()).device
        audio_tensor = torch.as_tensor(wav, device=device, dtype=torch.float32)
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        ctx = torch.no_grad() if not any(p.requires_grad for p in self.mimi.parameters()) else torch.enable_grad()
        with ctx:
            # wav is [C, T]; encode as batch of C mono streams, then flatten to codebooks.
            audio_tokens = self.mimi.encode(audio_tensor[:, None])

        audio_tokens = audio_tokens[..., : self.num_audio_frames]
        this_num_audio_frames = audio_tokens.shape[-1]
        audio_tokens = torch.nn.functional.pad(
            audio_tokens,
            (0, self.num_audio_frames - this_num_audio_frames),
            value=self.interleaver.zero_padding,
        )
        audio_tokens = audio_tokens.view(1, -1, self.num_audio_frames)
        if audio_tokens.shape[1] == 8:
            user = torch.full_like(audio_tokens, self.interleaver.zero_padding)
            audio_tokens = torch.cat([audio_tokens, user], dim=1)

        info_file = os.path.splitext(path)[0] + ".json"
        with open(info_file) as f:
            data = json.load(f)
            alignments = data["alignments"]

        start_alignment = _dicho(alignments, start_sec)
        end_alignment = _dicho(alignments, start_sec + self.duration_sec)
        alignments = [
            (a[0], (a[1][0] - start_sec, a[1][1] - start_sec), a[2])
            for a in alignments[start_alignment:end_alignment]
        ]

        text_tokens = self.interleaver.prepare_item(alignments, this_num_audio_frames)
        text_tokens = torch.nn.functional.pad(
            text_tokens,
            (0, self.num_audio_frames - text_tokens.shape[-1]),
            value=self.interleaver.zero_padding,
        )
        codes = torch.cat([text_tokens, audio_tokens], dim=1)
        prefix_frames = 0
        if self.hybrid.enabled and self._rng.random() < self.hybrid.proba:
            n_q = self.n_q if self.n_q is not None else audio_tokens.shape[1]
            dep_q = self.dep_q if self.dep_q is not None else min(8, n_q)
            info_prompt = data.get("system_prompt")
            prompt_text = info_prompt if info_prompt else self.hybrid.sample_system_prompt(self._rng)
            tok_ids = self.interleaver.tokenizer.encode(wrap_with_system_tags(prompt_text))
            if isinstance(tok_ids, list) and tok_ids and isinstance(tok_ids[0], list):
                tok_ids = tok_ids[0]
            prefix = build_hybrid_system_prefix(
                list(tok_ids),
                n_q=n_q,
                dep_q=dep_q,
                pad_id=self.hybrid.pad_id,
                silence_frames=self.hybrid.silence_frames(self.mimi.frame_rate),
                voice_codes=self._voice_codes(dep_q),
                device=codes.device,
            )
            prefix_frames = prefix.shape[-1]
            if prefix_frames > 0:
                keep = max(0, codes.shape[-1] - prefix_frames)
                codes = torch.cat([prefix.to(codes.device), codes[..., :keep]], dim=-1)
                if codes.shape[-1] < self.num_audio_frames:
                    codes = torch.nn.functional.pad(
                        codes,
                        (0, self.num_audio_frames - codes.shape[-1]),
                        value=self.interleaver.zero_padding,
                    )
                elif codes.shape[-1] > self.num_audio_frames:
                    codes = codes[..., : self.num_audio_frames]
                    prefix_frames = min(prefix_frames, self.num_audio_frames)
        return Sample(codes, data.get("text_conditions", None), prefix_frames=prefix_frames)
