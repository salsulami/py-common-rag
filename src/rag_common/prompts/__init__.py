"""Prompt-related utilities and defaults."""

from rag_common.prompts.defaults import (
    DEFAULT_PROMPTS,
    PROMPT_DOCX_CLEANUP,
    PROMPT_HYPOTHESIS,
    PROMPT_PDF_CLEANUP,
    PROMPT_PPTX_CLEANUP,
    PROMPT_QUERY_DECOMPOSITION,
    PROMPT_QUERY_EXPANSION,
)
from rag_common.prompts.registry import PromptRegistry

__all__ = [
    "DEFAULT_PROMPTS",
    "PROMPT_DOCX_CLEANUP",
    "PROMPT_HYPOTHESIS",
    "PROMPT_PDF_CLEANUP",
    "PROMPT_PPTX_CLEANUP",
    "PROMPT_QUERY_DECOMPOSITION",
    "PROMPT_QUERY_EXPANSION",
    "PromptRegistry",
]
