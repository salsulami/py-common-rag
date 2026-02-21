"""Prompt registry with update and override support."""

from __future__ import annotations

from collections.abc import Mapping

from rag_common.exceptions import PromptNotFoundError
from rag_common.prompts.defaults import DEFAULT_PROMPTS


class PromptRegistry:
    """Stores and serves prompt templates by key."""

    def __init__(self, prompts: Mapping[str, str] | None = None) -> None:
        self._prompts: dict[str, str] = dict(DEFAULT_PROMPTS)
        if prompts:
            self._prompts.update(prompts)

    def get(self, key: str) -> str:
        if key not in self._prompts:
            raise PromptNotFoundError(f"Unknown prompt key: {key}")
        return self._prompts[key]

    def set(self, key: str, prompt: str) -> None:
        if not prompt.strip():
            raise ValueError("Prompt must not be empty.")
        self._prompts[key] = prompt

    def update(self, prompts: Mapping[str, str]) -> None:
        for key, prompt in prompts.items():
            self.set(key, prompt)

    def clone(self) -> "PromptRegistry":
        return PromptRegistry(self._prompts)

    def as_dict(self) -> dict[str, str]:
        return dict(self._prompts)
