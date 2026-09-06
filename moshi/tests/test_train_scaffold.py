# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn

from moshi.architecture import MoshiSystem, apply_freeze
from moshi.models import lm as lm_mod
from moshi.train.loss import compute_loss_with_mask, moshi_loss
from moshi.train.data.interleaver import Interleaver


class _FakeSPM:
    def encode(self, text, enable_sampling=False, alpha=None, nbest_size=None):
        if isinstance(text, list):
            return [[11, 12] for _ in text]
        word = text if isinstance(text, str) else str(text)
        return [11] if word else []

    def bos_id(self):
        return 1

    def eos_id(self):
        return 2


class _FakeMimi(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.encoder_transformer = nn.Linear(4, 4)
        self.quantizer = nn.Module()
        self.quantizer.rvq_first = nn.Linear(4, 4)
        self.quantizer.rvq_rest = nn.Linear(4, 4)
        self.decoder_transformer = nn.Linear(4, 4)
        self.decoder = nn.Linear(4, 4)
        self.frame_rate = 12.5
        self.sample_rate = 24000


def _tiny_lm():
    return lm_mod.LMModel(
        delays=[0, 0, 1, 1],
        n_q=3,
        dep_q=3,
        card=32,
        text_card=48,
        dim=16,
        num_layers=1,
        num_heads=1,
        hidden_scale=1,
        depformer_dim=16,
        depformer_multi_linear=True,
        depformer_weights_per_step=True,
        depformer_num_heads=1,
        depformer_gating="silu",
        context=4,
        dtype=torch.float32,
    )


def test_freeze_map_mimi_not_temporal():
    system = MoshiSystem(_FakeMimi(), _tiny_lm())
    apply_freeze(system, {"mimi": True, "lm.temporal": False})
    assert not system.mimi.encoder.weight.requires_grad
    assert not system.mimi.quantizer.rvq_first.weight.requires_grad
    assert any(p.requires_grad for p in system.lm.transformer.parameters())
    assert any(p.requires_grad for p in system.lm.depformer.parameters())


def test_swap_encoder():
    system = MoshiSystem(_FakeMimi(), _tiny_lm())
    replacement = nn.Linear(4, 4)
    old = system.swap("mimi.encoder", replacement)
    assert system.component("mimi.encoder") is replacement
    assert isinstance(old, nn.Linear)


def test_token_layout_tiny_lm():
    model = _tiny_lm()
    assert model.num_codebooks == model.n_q + 1
    assert model.audio_offset == 1


def test_eq7_semantic_weight():
    B, K, T, card = 2, 3, 4, 8
    logits = torch.zeros(B, K, T, card)
    target = torch.zeros(B, K, T, dtype=torch.long)
    mask = torch.ones(B, K, T, dtype=torch.bool)
    logits[:, 0, :, 0] = 10.0
    weighted = compute_loss_with_mask(
        logits, target, mask, mode="audio", first_codebook_weight_multiplier=100.0
    )
    unweighted = compute_loss_with_mask(
        logits, target, mask, mode="audio", first_codebook_weight_multiplier=1.0
    )
    assert torch.isfinite(weighted)
    assert torch.isfinite(unweighted)


def test_moshi_loss_text_plus_audio():
    B, T, card, tcard = 1, 3, 8, 10
    text_logits = torch.zeros(B, 1, T, tcard)
    text_logits[..., 0] = 5.0
    text_target = torch.zeros(B, 1, T, dtype=torch.long)
    text_mask = torch.ones(B, 1, T, dtype=torch.bool)
    audio_logits = torch.zeros(B, 2, T, card)
    audio_logits[..., 0] = 5.0
    audio_target = torch.zeros(B, 2, T, dtype=torch.long)
    audio_mask = torch.ones(B, 2, T, dtype=torch.bool)
    total, text_loss, audio_loss = moshi_loss(
        text_logits, text_target, text_mask,
        audio_logits, audio_target, audio_mask,
        text_padding_ids={3, 0},
    )
    assert total.shape == ()
    assert text_loss.item() >= 0
    assert audio_loss.item() >= 0


def test_inner_monologue_epad():
    tok = _FakeSPM()
    inter = Interleaver(
        tok,
        audio_frame_rate=12.5,
        text_padding=3,
        end_of_text_padding=0,
        zero_padding=-1,
        keep_main_only=True,
        device="cpu",
    )
    alignments = [("hello", (0.08, 0.24), "SPEAKER_MAIN")]
    stream = inter.prepare_item(alignments, segment_duration=0.4)
    assert stream.shape[0] == 1 and stream.shape[1] == 1
    tokens = stream[0, 0].tolist()
    assert 0 in tokens  # EPAD
    assert 11 in tokens  # word piece
    assert 3 in tokens  # PAD
