"""Hash embedding used by tests. Not a semantic model."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pixeltable as pxt
from PIL import Image

DIM = 8


@pxt.udf
def tiny_embed(text: str) -> pxt.Array[(8,), pxt.Float]:
    digest = hashlib.sha256(text.encode()).digest()
    values = np.array([(digest[i % 32] / 127.5) - 1.0 for i in range(DIM)], dtype=np.float32)
    norm = float(np.linalg.norm(values))
    return values / norm if norm else values


@pxt.udf
def tiny_image_embed(image: Image.Image) -> pxt.Array[(8,), pxt.Float]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=(0, 1))
    values = np.pad(rgb, (0, DIM - 3))
    norm = float(np.linalg.norm(values))
    return values / norm if norm else values


@pxt.udf
def observe_read(text: str, marker: str) -> str:
    Path(marker).touch()
    return text
