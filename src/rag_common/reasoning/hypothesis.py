"""Hypothesis generation for RAG reasoning pipelines."""

from __future__ import annotations

from typing import Any, Sequence

from rag_common.llm import OpenAIClientAdapter
from rag_common.prompts import PROMPT_HYPOTHESIS, PromptRegistry
from rag_common.types import HypothesisCandidate, HypothesisSet


class HypothesisGenerator:
    """Produces evidence-seeking hypotheses from a query and retrieved context."""

    prompt_key = PROMPT_HYPOTHESIS

    def __init__(
        self,
        openai_client: Any,
        *,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.2,
        max_output_tokens: int = 1500,
        prompt_registry: PromptRegistry | None = None,
        prompt_override: str | None = None,
        llm_adapter_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._prompts = prompt_registry if prompt_registry else PromptRegistry()
        if prompt_override:
            self._prompts.set(self.prompt_key, prompt_override)
        llm_adapter_kwargs = dict(llm_adapter_kwargs or {})
        try:
            self._llm = OpenAIClientAdapter(
                openai_client,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                **llm_adapter_kwargs,
            )
        except TypeError as exc:
            raise ValueError("Invalid llm_adapter_kwargs for HypothesisGenerator.") from exc

    def update_default_prompt(self, prompt: str) -> None:
        """Update the default prompt used by this component."""
        self._prompts.set(self.prompt_key, prompt)

    def runtime_diagnostics(self) -> dict[str, Any]:
        """Return operational diagnostics for this component."""
        return self._llm.runtime_config()

    def generate(
        self,
        query: str,
        *,
        retrieved_context: Sequence[str],
        hypothesis_count: int = 4,
    ) -> HypothesisSet:
        if not query.strip():
            raise ValueError("query must not be empty.")
        if not retrieved_context:
            raise ValueError("retrieved_context must not be empty.")

        context_text = "\n".join(
            f"[context_{idx}] {chunk.strip()}"
            for idx, chunk in enumerate(retrieved_context, start=1)
            if chunk.strip()
        )
        item_limit = max(hypothesis_count, 1)
        prompt = self._prompts.get(self.prompt_key).format(
            query=query.strip(),
            context=context_text,
            hypothesis_count=item_limit,
        )
        payload = self._llm.complete_json(
            system_prompt="You generate cautious and evidence-oriented hypotheses.",
            user_prompt=prompt,
        )

        hypotheses = _normalize_hypotheses(payload.get("hypotheses"), item_limit)

        return HypothesisSet(
            query=query,
            hypotheses=hypotheses,
            overall_risk=_safe_string(payload.get("overall_risk")),
            raw_response=payload,
        )


def _normalize_hypotheses(value: Any, limit: int) -> list[HypothesisCandidate]:
    if not isinstance(value, list):
        return []

    normalized: list[HypothesisCandidate] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        statement = _safe_string(item.get("statement"))
        if not statement:
            continue
        confidence = _to_confidence(item.get("confidence"))
        raw_evidence = item.get("required_evidence", [])
        if not isinstance(raw_evidence, list):
            raw_evidence = []
        required_evidence = [
            entry.strip() for entry in raw_evidence if isinstance(entry, str) and entry.strip()
        ]
        rationale = _safe_string(item.get("rationale"))
        normalized.append(
            HypothesisCandidate(
                statement=statement,
                confidence=confidence,
                required_evidence=required_evidence,
                rationale=rationale,
            )
        )
        if limit > 0 and len(normalized) >= limit:
            break
    return normalized


def _safe_string(value: Any) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    return None


def _to_confidence(value: Any) -> float:
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        try:
            numeric = float(value.strip())
        except ValueError:
            return 0.0
    else:
        return 0.0
    if numeric < 0:
        return 0.0
    if numeric > 1:
        return 1.0
    return numeric
