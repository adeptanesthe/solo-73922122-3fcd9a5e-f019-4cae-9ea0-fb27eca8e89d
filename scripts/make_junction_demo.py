"""Self-contained synthetic sketch for the junction-check demo.

The drawing has three marked four-way crossings and two terminal pairs.  One
pair can only be joined through the top detour and the bottom path, which pins
two crossings; the third is not exercised by any record and stays uncertain --
exactly the "which crossings still need the original" answer the service
exists to give.

Usage:  python scripts/make_junction_demo.py [outdir]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

INK = 0
PAPER = 255


def _h(img: np.ndarray, y: int, x0: int, x1: int) -> None:
    img[y, x0 : x1 + 1] = INK


def _v(img: np.ndarray, x: int, y0: int, y1: int) -> None:
    img[y0 : y1 + 1, x] = INK


def _plus(img: np.ndarray, x: int, y: int, reach: int) -> None:
    _h(img, y, x - reach, x + reach)
    _v(img, x, y - reach, y + reach)


def sketch() -> np.ndarray:
    """80x80 single-pixel-line sketch with three marked crossings."""
    img = np.full((80, 80), PAPER, dtype=np.uint8)
    # Crossing A=(24,40) and B=(56,40): horizontal arms, with the direct span
    # between them cut so the only route is up and over the y=12 detour.
    _h(img, 40, 4, 25)
    _h(img, 40, 55, 76)
    _v(img, 24, 12, 64)
    _v(img, 56, 12, 64)
    _h(img, 12, 24, 56)
    # Crossing C=(40,64): alone on a plus, untouched by any terminal record.
    _plus(img, 40, 64, reach=10)
    return img


CROSSINGS = [(24, 40), (56, 40), (40, 64)]

TERMINALS = [
    # P=(8,40) and Q=(72,40) must be joined: only possible via the detour, so
    # both A and B are junctions.
    {"a": {"x": 8, "y": 40}, "b": {"x": 72, "y": 40}, "connected": True},
    # Dots on C's horizontal arms are always joined whatever C is; this record
    # is consistent but says nothing about C.
    {"a": {"x": 34, "y": 64}, "b": {"x": 46, "y": 64}, "connected": True},
]


def png_bytes(arr: np.ndarray) -> bytes:
    import io

    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def main(argv: list[str]) -> int:
    outdir = Path(argv[1] if len(argv) > 1 else "demo/junctions")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "sketch.png").write_bytes(png_bytes(sketch()))
    spec = {"crossings": [{"x": x, "y": y} for x, y in CROSSINGS], "terminals": TERMINALS}
    (outdir / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    cmd = (
        "curl -sS -X POST http://localhost:8000/junctions \\\n"
        f"  -F 'image=@{outdir}/sketch.png;type=image/png' \\\n"
        f"  -F 'spec=<{outdir / 'spec.json'}'"
    )
    (outdir / "request.sh").write_text(cmd + "\n", encoding="utf-8")
    print(f"wrote sketch.png, spec.json, request.sh to {outdir}/")
    print(f"then run: bash {outdir}/request.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
