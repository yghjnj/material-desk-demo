"""Module C: deterministic offline document retrieval and extractive evidence."""

from .core import (
    CConfig,
    CDocument,
    CResponse,
    DeterministicRetriever,
    ParsedBlock,
    canonicalize,
    parse_document,
)

__all__ = [
    "CConfig", "CDocument", "CResponse", "DeterministicRetriever",
    "ParsedBlock", "canonicalize", "parse_document",
]
