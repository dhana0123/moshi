# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""LLM components"""

from .utils import (
    get_llm,
    load_reference_prompt_template,
    REFERENCE_PROMPT_TEMPLATE_FILE,
    REFERENCE_PROMPT_TEMPLATE_FILE_SIMPLIFIED,
    REFERENCE_SUMMARIZATION_PROMPT_TEMPLATE_FILE,
)
from .client import LLMClient

__all__ = [
    "LLMClient",
    "get_llm",
    "load_reference_prompt_template",
    "REFERENCE_PROMPT_TEMPLATE_FILE",
    "REFERENCE_PROMPT_TEMPLATE_FILE_SIMPLIFIED",
    "REFERENCE_SUMMARIZATION_PROMPT_TEMPLATE_FILE",
]
