"""Local embedding functions for a Pixeltable memory index."""

from __future__ import annotations

import pixeltable as pxt

# Small local model. sentence-transformers resolves this id without an API key.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def sentence_transformer_embedding(
    model_id: str = DEFAULT_EMBEDDING_MODEL,
    *,
    normalize_embeddings: bool = True,
) -> pxt.Function:
    """Return a Pixeltable UDF that embeds text with a local sentence-transformers model.

    The first call loads the model (and downloads it if it is not cached). Install the
    weights stack with ``pip install 'strands-pixeltable[local]'``.

    The result is a ``.using()`` binding, so it can be stored in an embedding index.
    A function defined in a notebook cell or with ``.apply()`` cannot.
    """
    from pixeltable.functions.huggingface import sentence_transformer

    try:
        embedding = sentence_transformer.using(model_id=model_id, normalize_embeddings=normalize_embeddings)
    except (pxt.Error, ImportError, ModuleNotFoundError, OSError) as exc:
        raise RuntimeError(
            f"Could not load the local sentence-transformers model {model_id!r}. "
            "Install it with pip install 'strands-pixeltable[local]', or pass embedding= "
            "to PixeltableMemoryStore."
        ) from exc
    return embedding
