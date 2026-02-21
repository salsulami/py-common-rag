"""Custom exceptions for the RAG common library."""


class RagCommonError(Exception):
    """Base class for library exceptions."""


class PromptNotFoundError(RagCommonError):
    """Raised when a prompt key is unknown."""


class ResponseFormatError(RagCommonError):
    """Raised when an LLM response cannot be parsed as expected."""


class MissingDependencyError(RagCommonError):
    """Raised when an optional parser dependency is missing."""


class DocumentConversionError(RagCommonError):
    """Raised when a source file cannot be converted for visual rendering."""


class VisualExtractionErrorLimitExceeded(RagCommonError):
    """Raised when visual extraction exceeds the configured error budget."""


class UnsupportedFileTypeError(RagCommonError):
    """Raised when no parser supports a file extension."""
