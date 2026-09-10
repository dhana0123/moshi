# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from .client import LLMClient

REFERENCE_PROMPT_TEMPLATE_FILE = Path(__file__).parent / "reference_prompt_template.txt"
REFERENCE_PROMPT_TEMPLATE_FILE_SIMPLIFIED = Path(__file__).parent / "reference_prompt_template_simplified.txt"
REFERENCE_SUMMARIZATION_PROMPT_TEMPLATE_FILE = Path(__file__).parent / "reference_summarization_prompt_template.txt"


def get_llm(
    system_prompt: str,
    prompt_type: str,
    *,
    prompt_style: str = "original",
) -> LLMClient:
    if prompt_type == "empty":
        prompt = ""
    elif prompt_type == "reference":
        ref_path = (
            REFERENCE_PROMPT_TEMPLATE_FILE_SIMPLIFIED
            if prompt_style == "simplified"
            else REFERENCE_PROMPT_TEMPLATE_FILE
        )
        prompt = load_reference_prompt_template(filename=str(ref_path))
    elif prompt_type == "reference_summarization":
        prompt = load_reference_prompt_template(filename=str(REFERENCE_SUMMARIZATION_PROMPT_TEMPLATE_FILE))
    else:
        raise ValueError(f"Invalid prompt type: {prompt_type}")
    from .client import LLMClient

    return LLMClient(system_prompt=system_prompt, prompt=prompt)


def load_reference_prompt_template(
    filename: str = str(REFERENCE_PROMPT_TEMPLATE_FILE),
    **kwargs,
) -> str:
    """Load a prompt template and fill variables."""
    with open(filename) as f:
        template = f.read()
    return template
