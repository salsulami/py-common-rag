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
