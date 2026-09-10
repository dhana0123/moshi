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


def test_wrap_with_system_tags():
    from moshi.models.hybrid_prompt import wrap_with_system_tags

    assert wrap_with_system_tags("hello") == "<system> hello <system>"
    tagged = "<system> already <system>"
    assert wrap_with_system_tags(tagged) == tagged


def test_hybrid_prefix_layout():
    from moshi.models.hybrid_prompt import (
        SILENCE_TOKENS,
        SINE_TOKENS,
        apply_prefix_loss_mask,
        build_hybrid_system_prefix,
    )

    voice = torch.arange(8 * 3).view(8, 3)
    prefix = build_hybrid_system_prefix(
        [11, 12],
        n_q=16,
        dep_q=8,
        pad_id=3,
        silence_frames=2,
        voice_codes=voice,
    )
    # voice 3 + sil 2 + text 2 + sil 2
    assert prefix.shape == (1, 17, 9)
    assert (prefix[0, 0, :3] == 3).all()
    assert torch.equal(prefix[0, 1:9, :3], voice)
    sine = torch.as_tensor(SINE_TOKENS, dtype=torch.long)
    assert torch.equal(prefix[0, 9:, :3], sine.view(8, 1).expand(8, 3))
    assert (prefix[0, 0, 5:7] == torch.tensor([11, 12])).all()
    sil = torch.as_tensor(SILENCE_TOKENS, dtype=torch.long)
    assert torch.equal(prefix[0, 1:9, 5:7], sil.view(8, 1).expand(8, 2))

    text_mask = torch.ones(2, 1, 5, dtype=torch.bool)
    audio_mask = torch.ones(2, 3, 5, dtype=torch.bool)
    tmask, amask = apply_prefix_loss_mask(text_mask, audio_mask, torch.tensor([2, 0]))
    assert tmask[0, 0, :2].sum() == 0
    assert tmask[0, 0, 2:].all()
    assert tmask[1].all()
    assert amask[0, :, :2].sum() == 0


@torch.no_grad()
def test_lmgen_forced_hybrid_step():
    model = _tiny_lm()
    model.eval()
    gen = lm_mod.LMGen(model, use_sampling=False, temp=1.0, temp_text=1.0)
    user = torch.zeros(1, 0, 1, dtype=torch.long)
    moshi = torch.zeros(1, 3, 1, dtype=torch.long)
    with gen.streaming(1):
        for _ in range(4):
            gen.step(user, moshi_tokens=moshi, text_token=3)
        state = gen._streaming_state
        assert state is not None
        assert int(state.offset_cpu) >= 4


def test_rag_token_id_and_streaming_sum():
    from moshi.conditioners.base import ConditionFuser

    model = _tiny_lm()
    assert model.rag_token_id is None
    rag_lm = lm_mod.LMModel(
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
        rag_token_id=4,
    )
    assert rag_lm.rag_token_id == 4

    fuser = ConditionFuser(fuse2cond={"sum": [], "cross": [], "prepend": [], "streaming_sum": ["reference_with_time"]})
    dummy = torch.zeros(1, 2, 16)
    mask = torch.ones(1, 2)
    from moshi.conditioners.base import ConditionType
    summed = fuser.get_streaming_sum({"reference_with_time": ConditionType(dummy, mask)})
    assert summed is not None
    assert summed.shape == (1, 2, 16)


