"""Strands Agents memory store and read-only Pixeltable tools."""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version

from strands_pixeltable.embeddings import DEFAULT_EMBEDDING_MODEL, sentence_transformer_embedding
from strands_pixeltable.store import PixeltableMemoryStore
from strands_pixeltable.tools import pixeltable_tools

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "PixeltableMemoryStore",
    "pixeltable_tools",
    "sentence_transformer_embedding",
]

try:
    __version__ = _pkg_version("strands-pixeltable")
except PackageNotFoundError:  # source checkout without install
    __version__ = "0.0.0"
