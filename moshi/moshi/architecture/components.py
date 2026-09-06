# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Stable names for Moshi research knobs.

Each name maps to an attribute path on :class:`MoshiSystem`. Keep these strings
stable; freeze YAML and swap() calls depend on them.
"""

# research name -> attribute path from MoshiSystem
COMPONENT_ATTRS: dict[str, str] = {
    "mimi": "mimi",
    "mimi.encoder": "mimi.encoder",
    "mimi.encoder_transformer": "mimi.encoder_transformer",
    "mimi.quantizer": "mimi.quantizer",
    "mimi.quantizer.semantic": "mimi.quantizer.rvq_first",
    "mimi.quantizer.acoustic": "mimi.quantizer.rvq_rest",
    "mimi.decoder_transformer": "mimi.decoder_transformer",
    "mimi.decoder": "mimi.decoder",
    "lm": "lm",
    "lm.text_emb": "lm.text_emb",
    "lm.audio_emb": "lm.emb",
    "lm.temporal": "lm.transformer",
    "lm.text_linear": "lm.text_linear",
    "lm.depformer_in": "lm.depformer_in",
    "lm.depformer_emb": "lm.depformer_emb",
    "lm.depformer": "lm.depformer",
    "lm.audio_heads": "lm.linears",
}

ALIASES = {
    "helium": "lm.temporal",
    "temporal": "lm.temporal",
    "depformer": "lm.depformer",
    "depth": "lm.depformer",
    "codec": "mimi",
}


def canonical_name(name: str) -> str:
    name = ALIASES.get(name, name)
    if name not in COMPONENT_ATTRS:
        known = ", ".join(COMPONENT_ATTRS)
        raise KeyError(f"Unknown component {name!r}. Known: {known}")
    return name
