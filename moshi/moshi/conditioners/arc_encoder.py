# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import typing as tp

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from xformers.ops.fmha import memory_efficient_attention
from xformers.ops.fmha.attn_bias import BlockDiagonalCausalMask, BlockDiagonalMask
from safetensors.torch import load_file
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from .base import _BaseTextConditioner, ConditionType, TokenizedText
from ..utils.autocast import TorchAutocast
from ..conditioners.text import length_to_mask


logger = logging.getLogger(__name__)


def precompute_freqs_cis(dim: int, end: int, theta: float, device: torch.device | None = None) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    freqs_cis_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if freqs_cis_k is None:
        freqs_cis_k = freqs_cis.clone()

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    freqs_cis_k = freqs_cis_k[:, None, :]

    xq_out = torch.view_as_real(xq_ * freqs_cis)
    xk_out = torch.view_as_real(xk_ * freqs_cis_k)

    return xq_out.type_as(xq).flatten(-2), xk_out.type_as(xk).flatten(-2)


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=dim)
    values = torch.repeat_interleave(values, repeats=repeats, dim=dim)
    return keys, values


def positions_from_sizes(sizes: tp.Iterable[int], device) -> torch.Tensor:
    import operator
    from functools import reduce

    return torch.tensor(
        reduce(operator.iadd, [list(range(s)) for s in sizes], []),
        dtype=torch.long,
        device=device,
    )


def split_integer(x: int, n: int) -> list[int]:
    if n > 0:
        base = x // n
        remainder = x % n
        result = [base] * n
        for i in range(remainder):
            result[i] += 1
        return result
    else:
        n = -n
        base = x // n
        remainder = x % n
        if remainder > 0:
            result = (base + 1) * [x // (base + 1)]
            for i in range(x % (base + 1)):
                result[i] += 1
        else:
            result = [n] * base
        assert sum(result) == x, f"Sum of result {sum(result)} must be equal to x {x} with n {n}"
    return result


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        head_dim: int,
        n_kv_heads: int,
    ):
        super().__init__()
        self.n_heads: int = n_heads
        self.head_dim: int = head_dim
        self.n_kv_heads: int = n_kv_heads
        self.repeats = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        other_kv: torch.Tensor | None = None,
        freqs_cis: torch.Tensor | None = None,
        freqs_cis_k: torch.Tensor | None = None,
        mask: BlockDiagonalMask | BlockDiagonalCausalMask | torch.Tensor | None = None,
    ) -> torch.Tensor:
        seqlen_sum, _ = x.shape

        if other_kv is None:
            other_kv = x.clone()

        kv_seqlen, _ = other_kv.shape
        xq, xk, xv = self.wq(x), self.wk(other_kv), self.wv(other_kv)
        xq = xq.view(seqlen_sum, self.n_heads, self.head_dim)

        xk = xk.view(kv_seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(kv_seqlen, self.n_kv_heads, self.head_dim)

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis, freqs_cis_k=freqs_cis_k)

        key, val = xk, xv
        key, val = repeat_kv(key, val, self.repeats, dim=1)

        xq, key, val = xq[None, ...], key[None, ...], val[None, ...]

        if memory_efficient_attention is None:
            raise ImportError("xformers is required for ArcEncoderConditioner")
        output = memory_efficient_attention(xq, key, val, mask)
        output = output.view(seqlen_sum, self.n_heads * self.head_dim)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        norm_eps: float,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.attention = Attention(
            dim=dim,
            n_heads=n_heads,
            head_dim=head_dim,
            n_kv_heads=n_kv_heads,
        )
        self.attention_norm = RMSNorm(dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(dim, eps=norm_eps)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        other_kv: torch.Tensor | None = None,
        freqs_cis_k: torch.Tensor | None = None,
        mask: BlockDiagonalCausalMask | BlockDiagonalMask | torch.Tensor | None = None,
    ) -> torch.Tensor:
        r = self.attention.forward(
            x=self.attention_norm(x),
            freqs_cis=freqs_cis,
            mask=mask,
            other_kv=None if other_kv is None else self.attention_norm(other_kv),
            freqs_cis_k=freqs_cis_k,
        )

        h = x + r
        r = self.feed_forward.forward(self.ffn_norm(h))
        out = h + r
        return out


class PoolingModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        comp_rate: int,
        seqlens: list[int] | None = None,
    ) -> tuple[torch.Tensor, list[int]]:
        new_seqlens = []
        pool_size = []

        if comp_rate != -1 and seqlens is not None:
            for embed_size in seqlens:
                compressed_embed_size = []
                if comp_rate == 0:
                    compressed_embed_size = [embed_size]
                elif comp_rate > 0 and embed_size // comp_rate == 0:
                    compressed_embed_size = [1] * embed_size
                elif comp_rate < -1 and embed_size // abs(comp_rate) == 0:
                    compressed_embed_size = [embed_size]
                else:
                    compressed_embed_size = split_integer(embed_size, comp_rate)
                pool_size.extend(compressed_embed_size)
                new_seqlens.append(len(compressed_embed_size))

            pool_mask = torch.block_diag(*[torch.ones(t) / t for t in pool_size]).to(device=x.device, dtype=x.dtype)
        else:
            new_seqlens = seqlens if seqlens is not None else []
            pool_mask = None

        queries = x if pool_mask is None else pool_mask @ x
        return queries, new_seqlens


class ArcEncoderTransformer(nn.Module):
    def __init__(
        self,
        checkpoint: bool = False,
        compression_rate: int = -4,
    ):
        super().__init__()

        self.vocab_size = 128256
        self.n_layers = 26
        self._precomputed_freqs_cis: torch.Tensor | None = None

        self.tok_embeddings = torch.nn.Embedding(128256, 3072)
        self.for_embedding = True
        self.compress_rates = [compression_rate]
        self.n_layers = 26
        self.start_compressing = self.n_layers - len(self.compress_rates)
        self.trained_layers = range(0, self.n_layers)
        self.causal = False
        self.pooling_module = PoolingModule()
        self.n_mem_tokens = 0
        self.mem_embeddings = None

        layers = []
        for i in range(self.n_layers):
            block = TransformerBlock(
                dim=3072,
                hidden_dim=8192,
                n_heads=24,
                n_kv_heads=8,
                head_dim=128,
                norm_eps=1e-5,
            )

            if checkpoint:
                raise ImportError("torch.distributed is required for checkpointing")
            layers.append(block)

        self.layers = nn.ModuleDict({str(i): layers[i] for i in range(self.n_layers)})

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def freqs_cis(self) -> torch.Tensor:
        try:
            device = next(iter(self.parameters())).device
        except StopIteration:
            device = torch.device("cuda")

        if self._precomputed_freqs_cis is None:
            theta = 500000.0
            self._precomputed_freqs_cis = precompute_freqs_cis(128, 128_000, theta=theta, device=device)

        return self._precomputed_freqs_cis

    def forward_embedder(
        self,
        input_ids: torch.Tensor,
        seqlens: list[int],
    ) -> tuple[torch.Tensor, list[int]]:
        assert sum(seqlens) == input_ids.shape[0], (sum(seqlens), input_ids.shape[0])
        token_embeds = self.tok_embeddings(input_ids)

        h = token_embeds
        positions = positions_from_sizes(seqlens, self.freqs_cis.device)

        if BlockDiagonalMask is None or BlockDiagonalCausalMask is None:
            raise ImportError("xformers is required for ArcEncoderConditioner")

        if not self.causal:
            self_att_mask = BlockDiagonalMask.from_seqlens(seqlens)
        else:
            self_att_mask = BlockDiagonalCausalMask.from_seqlens(seqlens)

        freqs_cis = self.freqs_cis[positions].to(device=h.device)
        compress_index = 0

        for i in range(self.n_layers):
            if not isinstance(self_att_mask, BlockDiagonalMask):
                self_att_mask = BlockDiagonalMask.from_seqlens(seqlens)

            if i >= self.start_compressing:
                pooled_h, new_seqlens = self.pooling_module(
                    x=h,
                    comp_rate=self.compress_rates[compress_index],
                    seqlens=seqlens,
                )
                positions = positions_from_sizes(new_seqlens, self.freqs_cis.device)
                new_freqs_cis = self.freqs_cis[positions].to(device=h.device)

                if not self.causal:
                    self_att_mask = BlockDiagonalMask.from_seqlens(q_seqlen=new_seqlens, kv_seqlen=seqlens)
                else:
                    self_att_mask = BlockDiagonalCausalMask.from_seqlens(q_seqlen=new_seqlens, kv_seqlen=seqlens)

                h = self.layers[str(i)](
                    x=pooled_h,
                    other_kv=h,
                    freqs_cis=new_freqs_cis,
                    mask=self_att_mask,
                    freqs_cis_k=freqs_cis,
                )

                if not self.causal:
                    self_att_mask = BlockDiagonalMask.from_seqlens(q_seqlen=new_seqlens, kv_seqlen=new_seqlens)
                else:
                    self_att_mask = BlockDiagonalCausalMask.from_seqlens(q_seqlen=new_seqlens, kv_seqlen=new_seqlens)
                freqs_cis = new_freqs_cis
                seqlens = new_seqlens
                compress_index += 1
            else:
                h = self.layers[str(i)](
                    x=h,
                    freqs_cis=freqs_cis,
                    mask=self_att_mask,
                )

        if self.n_mem_tokens > 0:
            new_h = torch.zeros(
                (self.n_mem_tokens * len(seqlens), h.shape[1]),
                device=h.device,
                dtype=h.dtype,
            )
            ind = 0
            for j, size in enumerate(seqlens):
                new_h[j * self.n_mem_tokens : (j + 1) * self.n_mem_tokens] = h[ind : ind + size][-self.n_mem_tokens :]
                ind += size
            seqlens = [self.n_mem_tokens] * len(seqlens)
            h = new_h.clone()

        return h, seqlens


class EmbProjector(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = out_dim

        self.layer1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.layer2 = nn.Linear(hidden_dim, out_dim, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ArcEncoderTokenizer:
    def __init__(self, model_name: str):
        logger.info(f"Loading tokenizer from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_idx = -1
        self.bos_token = self.tokenizer.bos_token_id
        self.eos_token = self.tokenizer.eos_token_id
        self.stop_tokens = {self.eos_token}

    def encode(
        self,
        text: str,
        *,
        bos: bool = False,
        eos: bool = False,
        allowed_special: tp.Union[str, tp.AbstractSet[str]] = set(),
        disallowed_special: tp.Union[str, tp.Collection[str]] = (),
    ) -> list[int]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if bos and self.bos_token is not None:
            tokens.insert(0, self.bos_token)
        if eos and self.eos_token is not None:
            tokens.append(self.eos_token)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)


class ArcEncoderConditioner(_BaseTextConditioner[TokenizedText]):
    def __init__(
        self,
        finetune: bool = False,
        autocast_dtype: tp.Optional[str] = "bfloat16",
        tokenizer_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        hf_repo: str | None = None,
        **kwargs,
    ):
        self._hf_repo: str | None = hf_repo
        self.finetune = finetune

        embedder_params = kwargs.pop("embedder_params", None)
        bridge_module = kwargs.pop("bridge_module", None)
        if embedder_params is None or bridge_module is None:
            raise ValueError("ArcEncoderConditioner: pass embedder_params and bridge_module.")
        self.config = {"embedder_params": embedder_params, "bridge_module": bridge_module}

        self.compression_rate = self.config["embedder_params"]["compress_rates"][0]
        self.bridge_module_params = self.config["bridge_module"]

        super().__init__(dim=self.bridge_module_params["out_dim"], **kwargs)

        if autocast_dtype is None or self.device == "cpu":
            self.autocast = TorchAutocast(enabled=False)
        else:
            dtype = getattr(torch, autocast_dtype)
            assert isinstance(dtype, torch.dtype)
            self.autocast = TorchAutocast(enabled=True, device_type=self.device.split(":")[0], dtype=dtype)

        self.tokenizer = ArcEncoderTokenizer(tokenizer_name)

        self._init_modules()

    def _init_modules(self):
        self.embedder = ArcEncoderTransformer(compression_rate=self.compression_rate)
        self.embedder.to(self.device)
        self.bridge_module = EmbProjector(
            in_dim=self.bridge_module_params["in_dim"],
            out_dim=self.bridge_module_params["out_dim"],
            hidden_dim=self.bridge_module_params["hidden_dim"],
        ).to(self.device)

        if self.finetune:
            self.embedder.train()
            self.bridge_module.train()
        else:
            self.embedder.eval()
            self.bridge_module.eval()

    def load_weights(self) -> None:
        """If ``hf_repo`` was set in the constructor, load ``model.safetensors`` from Hugging Face.

        Intended to be called from ``loaders.get_moshi_lm`` after the full Moshi checkpoint load.
        No-op if ``hf_repo`` was not set.
        """
        if not self._hf_repo:
            return

        path = hf_hub_download(self._hf_repo, "model.safetensors")
        state = load_file(path, device=str(self.device))
        self.load_state_dict(state, assign=True, strict=False)
        if self.finetune:
            self.embedder.train()
            self.bridge_module.train()
        else:
            self.embedder.eval()
            self.bridge_module.eval()

    def prepare(self, x: tp.List[tp.Optional[str]]) -> TokenizedText:
        entries: tp.List[str] = [xi if xi is not None else "" for xi in x]

        output, lengths = [], []
        for text in entries:
            if text == "":
                output.append(torch.tensor([self.tokenizer.pad_idx]))
                lengths.append(0)
                continue

            tokens = self.tokenizer.encode(text, bos=False, eos=False)
            lengths.append(len(tokens))
            output.append(torch.tensor(tokens))

        mask = length_to_mask(torch.tensor(lengths))
        padded_output = pad_sequence(output, padding_value=self.tokenizer.pad_idx, batch_first=True).int()

        return TokenizedText(padded_output.to(self.device), mask.to(self.device))

    def _get_condition(self, inputs: TokenizedText) -> ConditionType:
        tokens, mask = inputs
        batch_size, _ = tokens.shape
        assert batch_size == 1, "ArcEncoderConditioner only supports one reference for now"
        valid_tokens = tokens[0][mask[0]]

        if valid_tokens.shape[0] == 0:
            # All sequences are empty
            # Return a single zero vector with sequence length 1 to avoid error caused by empty tensor
            final_embeddings = torch.zeros(1, 1, self.bridge_module_params["out_dim"], device=self.device)
            final_mask = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
            return ConditionType(final_embeddings, final_mask)

        with torch.set_grad_enabled(False), self.autocast:
            embeddings, embed_seqlens = self.embedder.forward_embedder(
                input_ids=valid_tokens, seqlens=[valid_tokens.shape[0]]
            )

            embeddings = self.bridge_module(embeddings).unsqueeze(0)
            masks = torch.ones(1, embed_seqlens[0], dtype=torch.bool, device=self.device)

        return ConditionType(embeddings, masks)


class MultiArcEncoderConditioner(ArcEncoderConditioner):
    """ArcEncoder-based conditioner that handles multiple conditioning texts for one sample. This conditioner takes a list of reference texts.

    Args:
        finetune (bool): Whether to fine-tune the model at train time.
        autocast_dtype (str, optional): Autocast dtype for mixed precision.
        tokenizer_name (str): Name of the tokenizer to use.
        ref_dropout (float, optional): Reference dropout probability. Not useful at inference time.
        frame_rate (float, optional): Frame rate for converting time to frame indices. Defaults to 12.5. Not useful at inference time.
        rag_time_sampling_params (dict, optional): Parameters for RAG time sampling. Not useful at inference time.
        embedder_params (dict): Embedder architecture (e.g. ``compress_rates``).
        bridge_module (dict): Bridge ``in_dim`` / ``out_dim`` / ``hidden_dim``.
        hf_repo (str, optional): If set, ``loaders.get_moshi_lm`` loads merged ARC weights from this HF repo after the Moshi checkpoint.
    """

    def __init__(
        self,
        finetune: bool = False,
        autocast_dtype: tp.Optional[str] = "bfloat16",
        tokenizer_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        ref_dropout: float = 0.0,
        frame_rate: float = 12.5,
        rag_time_sampling_params: tp.Optional[tp.Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__(
            finetune=finetune,
            autocast_dtype=autocast_dtype,
            tokenizer_name=tokenizer_name,
            **kwargs,
        )

    def _get_condition(
        self,
        inputs: TokenizedText,
    ) -> ConditionType:
        """Get condition tensor from multiple references. But during inference time, we only process one reference at a time since this forward pass is very fast.

        Args:
            inputs: TokenizedText

        Returns:
            ConditionType containing condition and mask
        """
        embeddings, raw_mask = super()._get_condition(inputs)
        B = inputs.tokens.shape[0]
        assert B == 1, "MultiArcEncoderConditioner only supports one reference for now"

        # Check whether the reference is empty with input_mask since output_mask may still has non-zero values even if the input is an empty string
        if inputs.mask.sum() != 0:
            # raw_mask is all True if inputs is not empty
            return ConditionType(embeddings, raw_mask)
        else:
            # Set mask to False when the reference is empty since this is where we would like to apply the learnt padding if self.learn_padding is True
            return ConditionType(torch.zeros_like(embeddings), torch.zeros_like(raw_mask))
