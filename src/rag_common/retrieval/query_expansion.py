"""Query expansion component for retrieval pipelines."""

from __future__ import annotations

from typing import Any

from rag_common.llm import OpenAIClientAdapter
from rag_common.prompts import PROMPT_QUERY_EXPANSION, PromptRegistry
from rag_common.types import QueryExpansionResult


class QueryExpander:
    """Generates alternative retrieval-ready formulations of a user query."""

    prompt_key = PROMPT_QUERY_EXPANSION

    def __init__(
        self,
        openai_client: Any,
        *,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.2,
        max_output_tokens: int = 1200,
        prompt_registry: PromptRegistry | None = None,
        prompt_override: str | None = None,
    ) -> None:
        self._prompts = prompt_registry if prompt_registry else PromptRegistry()
        if prompt_override:
            self._prompts.set(self.prompt_key, prompt_override)
        self._llm = OpenAIClientAdapter(
            openai_client,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

    def update_default_prompt(self, prompt: str) -> None:
        """Update the default prompt used by this component."""
        self._prompts.set(self.prompt_key, prompt)

    def expand(
        self,
        query: str,
        *,
        context: str | None = None,
        target_count: int = 5,
        include_original: bool = True,
    ) -> QueryExpansionResult:
        if not query.strip():
            raise ValueError("query must not be empty.")

        item_limit = max(target_count, 1)
        prompt = self._prompts.get(self.prompt_key).format(
            query=query.strip(),
            context=(context or "N/A").strip(),
            target_count=item_limit,
        )
        payload = self._llm.complete_json(
            system_prompt="You create robust retrieval query expansions.",
            user_prompt=prompt,
        )

        expansions = _normalize_string_list(payload.get("expansions"))
        if not expansions:
            expansions = _normalize_string_list(payload.get("queries"))
        if include_original and query.strip() not in expansions:
            expansions.insert(0, query.strip())

        expansions = expansions[:item_limit]

        return QueryExpansionResult(
            original_query=query,
            expansions=expansions,
            rationale=_safe_string(payload.get("rationale")),
            raw_response=payload,
        )


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []

    result: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _safe_string(value: Any) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None
