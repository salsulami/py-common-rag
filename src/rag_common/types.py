"""Shared datatypes used by RAG components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


JSONDict = dict[str, Any]


@dataclass(slots=True)
class DocumentChunk:
    """Represents a chunk of parsed source text."""

    text: str
    order: int
    metadata: JSONDict = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDocument:
    """Represents a parsed source document."""

    source_path: str
    content: str
    chunks: list[DocumentChunk]
    metadata: JSONDict = field(default_factory=dict)


@dataclass(slots=True)
class QueryExpansionResult:
    """Normalized output of query expansion."""

    original_query: str
    expansions: list[str]
    rationale: str | None = None
    raw_response: JSONDict | None = None


@dataclass(slots=True)
class DecomposedQuery:
    """A single sub-query in a decomposition plan."""

    id: str
    query: str
    intent: str | None = None
    depends_on: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QueryDecompositionResult:
    """Normalized output of query decomposition."""

    original_query: str
    sub_queries: list[DecomposedQuery]
    execution_notes: str | None = None
    raw_response: JSONDict | None = None


@dataclass(slots=True)
class HypothesisCandidate:
    """A generated hypothesis for a user question."""

    statement: str
    confidence: float
    required_evidence: list[str] = field(default_factory=list)
    rationale: str | None = None


@dataclass(slots=True)
class HypothesisSet:
    """Normalized output of hypothesis generation."""

    query: str
    hypotheses: list[HypothesisCandidate]
    overall_risk: str | None = None
    raw_response: JSONDict | None = None


@dataclass(slots=True)
class VisualExtractionItem:
    """Single page/slide extraction result from screenshot + vision LLM."""

    index: int
    markdown: str
    caption: str
    metadata: JSONDict = field(default_factory=dict)
    raw_response: JSONDict | None = None

    def to_dict(self) -> JSONDict:
        payload: JSONDict = {
            "index": self.index,
            "markdown": self.markdown,
            "caption": self.caption,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class VisualExtractionError:
    """Captures a recoverable per-item extraction failure."""

    index: int
    item_type: str
    error: str
    metadata: JSONDict = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        payload: JSONDict = {
            "index": self.index,
            "item_type": self.item_type,
            "error": self.error,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class FileManifest:
    """Metadata describing the document-level extraction operation."""

    source_path: str
    file_type: str
    item_type: str
    item_count: int
    extraction_mode: str = "vision_markdown_caption"
    metadata: JSONDict = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        payload: JSONDict = {
            "source_path": self.source_path,
            "file_type": self.file_type,
            "item_type": self.item_type,
            "item_count": self.item_count,
            "extraction_mode": self.extraction_mode,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True)
class VisualExtractionResult:
    """Combined file manifest + extracted page/slide items."""

    file_manifest: FileManifest
    items: list[VisualExtractionItem]
    errors: list[VisualExtractionError] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        payload: JSONDict = {
            "file_manifest": self.file_manifest.to_dict(),
            "items": [item.to_dict() for item in self.items],
        }
        if self.errors:
            payload["errors"] = [error.to_dict() for error in self.errors]
        return payload
