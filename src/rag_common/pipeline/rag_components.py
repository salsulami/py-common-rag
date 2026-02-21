"""High-level assembly of common RAG components."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Sequence

from rag_common.parsers import DocxParser, DocumentParserRouter, PdfParser, PptxParser
from rag_common.prompts import PromptRegistry
from rag_common.reasoning import HypothesisGenerator
from rag_common.retrieval import QueryDecomposer, QueryExpander
from rag_common.types import (
    HypothesisSet,
    JSONDict,
    ParsedDocument,
    QueryDecompositionResult,
    QueryExpansionResult,
    VisualExtractionItem,
    VisualExtractionResult,
)


class RagComponents:
    """
    Unified holder for retrieval, reasoning, and parser components.

    All components share one prompt registry so updating prompts propagates everywhere.
    """

    def __init__(
        self,
        *,
        openai_client: Any,
        model: str = "gpt-4.1-mini",
        prompt_overrides: Mapping[str, str] | None = None,
        normalize_parsed_text_with_llm: bool = False,
        llm_adapter_kwargs: JSONDict | None = None,
    ) -> None:
        self.prompts = PromptRegistry(prompt_overrides)
        llm_adapter_kwargs = dict(llm_adapter_kwargs or {})

        self.query_expander = QueryExpander(
            openai_client=openai_client,
            model=model,
            prompt_registry=self.prompts,
            llm_adapter_kwargs=llm_adapter_kwargs,
        )
        self.query_decomposer = QueryDecomposer(
            openai_client=openai_client,
            model=model,
            prompt_registry=self.prompts,
            llm_adapter_kwargs=llm_adapter_kwargs,
        )
        self.hypothesis_generator = HypothesisGenerator(
            openai_client=openai_client,
            model=model,
            prompt_registry=self.prompts,
            llm_adapter_kwargs=llm_adapter_kwargs,
        )

        self.pdf_parser = PdfParser(
            openai_client=openai_client,
            model=model,
            prompt_registry=self.prompts,
            normalize_with_llm=normalize_parsed_text_with_llm,
            llm_adapter_kwargs=llm_adapter_kwargs,
        )
        self.pptx_parser = PptxParser(
            openai_client=openai_client,
            model=model,
            prompt_registry=self.prompts,
            normalize_with_llm=normalize_parsed_text_with_llm,
            llm_adapter_kwargs=llm_adapter_kwargs,
        )
        self.docx_parser = DocxParser(
            openai_client=openai_client,
            model=model,
            prompt_registry=self.prompts,
            normalize_with_llm=normalize_parsed_text_with_llm,
            llm_adapter_kwargs=llm_adapter_kwargs,
        )
        self.parser_router = DocumentParserRouter(
            pdf_parser=self.pdf_parser,
            pptx_parser=self.pptx_parser,
            docx_parser=self.docx_parser,
        )

    def update_prompt(self, key: str, prompt: str) -> None:
        """Update a shared prompt key used by components."""
        self.prompts.set(key, prompt)

    def set_visual_error_policy(self, *, continue_on_error: bool, max_visual_errors: int = 20) -> None:
        """Apply a shared visual extraction error policy to all document parsers."""
        self.pdf_parser.set_visual_error_policy(
            continue_on_error=continue_on_error,
            max_visual_errors=max_visual_errors,
        )
        self.pptx_parser.set_visual_error_policy(
            continue_on_error=continue_on_error,
            max_visual_errors=max_visual_errors,
        )
        self.docx_parser.set_visual_error_policy(
            continue_on_error=continue_on_error,
            max_visual_errors=max_visual_errors,
        )

    def expand_query(
        self,
        query: str,
        *,
        context: str | None = None,
        target_count: int = 5,
    ) -> QueryExpansionResult:
        return self.query_expander.expand(query, context=context, target_count=target_count)

    def decompose_query(
        self,
        query: str,
        *,
        context: str | None = None,
        max_steps: int = 6,
    ) -> QueryDecompositionResult:
        return self.query_decomposer.decompose(query, context=context, max_steps=max_steps)

    def generate_hypotheses(
        self,
        query: str,
        *,
        retrieved_context: Sequence[str],
        hypothesis_count: int = 4,
    ) -> HypothesisSet:
        return self.hypothesis_generator.generate(
            query,
            retrieved_context=retrieved_context,
            hypothesis_count=hypothesis_count,
        )

    def parse_document(self, source_path: str) -> ParsedDocument:
        return self.parser_router.parse(source_path)

    def stream_visual_items(
        self,
        source_path: str,
        *,
        dpi: int = 170,
    ) -> Iterator[VisualExtractionItem]:
        """Stream page/slide extraction results one item at a time."""
        yield from self.parser_router.stream_visual_items(source_path, dpi=dpi)

    def stream_visual_json_items(
        self,
        source_path: str,
        *,
        dpi: int = 170,
    ) -> Iterator[JSONDict]:
        """Stream item JSON payloads (`index`, `markdown`, `caption`) one by one."""
        yield from self.parser_router.stream_visual_json_items(source_path, dpi=dpi)

    def extract_visual_result(
        self,
        source_path: str,
        *,
        dpi: int = 170,
        continue_on_error: bool | None = None,
        max_errors: int | None = None,
    ) -> VisualExtractionResult:
        """Process an entire document and return manifest + all items."""
        return self.parser_router.extract_visual_result(
            source_path,
            dpi=dpi,
            continue_on_error=continue_on_error,
            max_errors=max_errors,
        )

    def extract_visual_json(
        self,
        source_path: str,
        *,
        dpi: int = 170,
        continue_on_error: bool | None = None,
        max_errors: int | None = None,
    ) -> JSONDict:
        """
        Process an entire document and return:
        {
          "file_manifest": {...},
          "items": [{"index": ..., "markdown": ..., "caption": ...}, ...]
        }
        """
        return self.parser_router.extract_visual_json(
            source_path,
            dpi=dpi,
            continue_on_error=continue_on_error,
            max_errors=max_errors,
        )

    def runtime_diagnostics(self) -> JSONDict:
        """Return operational diagnostics for deployment checks."""
        return {
            "parsers": self.parser_router.runtime_diagnostics(),
            "retrieval": {
                "query_expander": self.query_expander.runtime_diagnostics(),
                "query_decomposer": self.query_decomposer.runtime_diagnostics(),
            },
            "reasoning": {
                "hypothesis_generator": self.hypothesis_generator.runtime_diagnostics(),
            },
        }
