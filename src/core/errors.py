class RagError(Exception):
    """Base class for errors this app raises on purpose, with a user-facing message."""


class DocumentParseError(RagError):
    """PDF could not be parsed — encrypted, corrupted, or an unsupported/non-PDF file."""


class EmptyDocumentError(RagError):
    """PDF parsed successfully but produced zero usable chunks (blank pages, no text/tables/pictures)."""


class MissingAPIKeyError(RagError):
    """OPENAI_API_KEY is not set in the environment."""
