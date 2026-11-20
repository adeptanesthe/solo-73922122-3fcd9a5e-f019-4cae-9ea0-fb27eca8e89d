"""Demo client for ``POST /junctions`` -- no Docker required.

    uvicorn app.main:app --port 8000 &
    python scripts/junction_demo_client.py --base-url http://localhost:8000

Exercises the real HTTP surface (multipart upload + JSON spec).  Exits 0 when
the service answers with a non-conflict verdict, 1 otherwise, 2 on transport
failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_junction_demo import CROSSINGS, TERMINALS, png_bytes, sketch  # noqa: E402


def build_multipart(image: bytes, spec: str) -> tuple[bytes, str]:
    boundary = "----junctiondemo"
    parts = [
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; '
            f'filename="sketch.png"\r\nContent-Type: image/png\r\n\r\n'
        ).encode()
        + image
        + b"\r\n",
        (
            f'--{boundary}\r\nContent-Disposition: form-data; name="spec"\r\n\r\n{spec}\r\n'
        ).encode(),
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    args = parser.parse_args()

    spec = json.dumps(
        {
            "crossings": [{"x": x, "y": y} for x, y in CROSSINGS],
            "terminals": TERMINALS,
        }
    )
    body, content_type = build_multipart(png_bytes(sketch()), spec)
    req = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/junctions",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.load(resp)
    except Exception as exc:  # noqa: BLE001 - demo tool
        print(f"request failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload.get("status") in ("resolved", "needs_review") else 1


if __name__ == "__main__":
    raise SystemExit(main())
