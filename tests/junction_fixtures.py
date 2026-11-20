"""Synthetic line sketches for junction-check tests."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

INK = 0
PAPER = 255


def blank(size: int = 64) -> np.ndarray:
    return np.full((size, size), PAPER, dtype=np.uint8)


def hline(img: np.ndarray, y: int, x0: int, x1: int) -> None:
    """Draw ink pixels from x0 to x1 inclusive on row y."""
    img[y, x0 : x1 + 1] = INK


def vline(img: np.ndarray, x: int, y0: int, y1: int) -> None:
    """Draw ink pixels from y0 to y1 inclusive on column x."""
    img[y0 : y1 + 1, x] = INK


def png_bytes(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def png_bytes_rgb(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.stack([img] * 3, axis=-1), mode="RGB").save(buf, format="PNG")
    return buf.getvalue()
