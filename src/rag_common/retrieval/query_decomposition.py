"""Query decomposition component for multi-hop retrieval."""

from __future__ import annotations

from typing import Any

from rag_common.llm import OpenAIClientAdapter
from rag_common.prompts import PROMPT_QUERY_DECOMPOSITION, PromptRegistry
from rag_common.types import DecomposedQuery, QueryDecompositionResult


class QueryDecomposer:
    """Breaks complex questions into dependency-aware sub-queries."""

    prompt_key = PROMPT_QUERY_DECOMPOSITION

    def __init__(
        self,
        openai_client: Any,
        *,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.2,
        max_output_tokens: int = 1400,
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

    def decompose(
        self,
        query: str,
        *,
        context: str | None = None,
        max_steps: int = 6,
    ) -> QueryDecompositionResult:
        if not query.strip():
            raise ValueError("query must not be empty.")

        step_limit = max(max_steps, 1)
        prompt = self._prompts.get(self.prompt_key).format(
            query=query.strip(),
            context=(context or "N/A").strip(),
            max_steps=step_limit,
        )
        payload = self._llm.complete_json(
            system_prompt="You decompose complex user questions for retrieval planning.",
            user_prompt=prompt,
        )

        sub_queries = _normalize_sub_queries(payload.get("sub_queries"), max_steps=step_limit)

        return QueryDecompositionResult(
            original_query=query,
            sub_queries=sub_queries,
            execution_notes=_safe_string(payload.get("execution_notes")),
            raw_response=payload,
        )


def _normalize_sub_queries(value: Any, *, max_steps: int) -> list[DecomposedQuery]:
    if not isinstance(value, list):
        return []

    normalized: list[DecomposedQuery] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        sub_query = _parse_sub_query(item=item, fallback_index=index + 1)
        if sub_query is None:
            continue
        normalized.append(sub_query)
        if max_steps > 0 and len(normalized) >= max_steps:
            break
    return normalized


def _parse_sub_query(item: dict[str, Any], fallback_index: int) -> DecomposedQuery | None:
    query = _safe_string(item.get("query"))
    if not query:
        return None

    query_id = _safe_string(item.get("id")) or f"q{fallback_index}"
    intent = _safe_string(item.get("intent"))
    raw_dependencies = item.get("depends_on", [])
    if not isinstance(raw_dependencies, list):
        raw_dependencies = []
    depends_on = [dep.strip() for dep in raw_dependencies if isinstance(dep, str) and dep.strip()]
    return DecomposedQuery(id=query_id, query=query, intent=intent, depends_on=depends_on)


def _safe_string(value: Any) -> str | None:
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None
