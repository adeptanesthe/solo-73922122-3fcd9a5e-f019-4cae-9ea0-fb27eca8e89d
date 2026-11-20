"""End-to-end tests for ``POST /junctions``."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.junction_fixtures import blank, hline, png_bytes, png_bytes_rgb, vline

client = TestClient(app)


def plus(img: np.ndarray, x: int, y: int, reach: int = 12) -> None:
    hline(img, y, x - reach, x + reach)
    vline(img, x, y - reach, y + reach)


def post(img: np.ndarray, spec: dict, *, filename="sketch.png", content_type="image/png"):
    return client.post(
        "/junctions",
        files={"image": (filename, png_bytes(img), content_type)},
        data={"spec": json.dumps(spec)},
    )


def pt(x: int, y: int) -> dict:
    return {"x": x, "y": y}


def rec(a, b, connected) -> dict:
    return {"a": pt(*a), "b": pt(*b), "connected": connected}


def test_docs_lists_new_endpoint():
    schema = client.get("/openapi.json").json()
    assert "/junctions" in schema["paths"]
    assert client.get("/docs").status_code == 200


def test_forced_junction():
    img = blank()
    plus(img, 32, 32)
    r = post(
        img,
        {"crossings": [pt(32, 32)], "terminals": [rec((20, 32), (32, 20), True)]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "resolved"
    assert body["consistent_assignments"] == 1
    assert body["crossings"] == [{"x": 32, "y": 32, "state": "junction"}]


def test_forced_crossover():
    img = blank()
    plus(img, 32, 32)
    r = post(
        img,
        {"crossings": [pt(32, 32)], "terminals": [rec((20, 32), (32, 20), False)]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "resolved"
    assert body["crossings"][0]["state"] == "crossover"


def test_uncertain_when_only_straight_record_is_given():
    img = blank()
    plus(img, 32, 32)
    r = post(
        img,
        {"crossings": [pt(32, 32)], "terminals": [rec((20, 32), (44, 32), True)]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "needs_review"
    assert body["crossings"][0]["state"] == "uncertain"
    assert body["consistent_assignments"] == 2


def test_conflict_reports_no_verdicts():
    img = blank()
    plus(img, 32, 32)
    r = post(
        img,
        {
            "crossings": [pt(32, 32)],
            "terminals": [
                rec((32, 20), (32, 44), False),
                rec((20, 32), (32, 20), False),
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "conflict"
    assert body["crossings"] == []
    assert body["consistent_assignments"] == 0
    assert "冲突" in body["message"]


def test_verdicts_preserve_request_order():
    img = blank(size=96)
    plus(img, 20, 20)
    plus(img, 70, 70)
    r = post(
        img,
        {
            "crossings": [pt(70, 70), pt(20, 20)],
            "terminals": [rec((8, 20), (20, 8), True)],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert [c["x"] for c in body["crossings"]] == [70, 20]
    assert body["crossings"][1]["state"] == "junction"
    assert body["crossings"][0]["state"] == "uncertain"


def test_route_through_other_crossings_is_considered():
    # P and Q are joined only via the top detour through crossings A and B;
    # both must be junctions for the record to hold.  This is the "path through
    # other crossings" requirement.
    img = blank()
    hline(img, 32, 4, 17)
    hline(img, 32, 47, 60)
    vline(img, 16, 8, 56)
    vline(img, 48, 8, 56)
    hline(img, 8, 16, 48)
    r = post(
        img,
        {
            "crossings": [pt(16, 32), pt(48, 32)],
            "terminals": [rec((6, 32), (58, 32), True), rec((6, 32), (58, 32), True)],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "resolved"
    assert [c["state"] for c in body["crossings"]] == ["junction", "junction"]


def test_rejects_sketch_over_256_px():
    img = blank(size=257)
    plus(img, 128, 128)
    r = post(img, {"crossings": [pt(128, 128)], "terminals": [rec((120, 128), (128, 120), True)]})
    assert r.status_code == 400
    assert "256" in r.json()["detail"]


def test_accepts_256_px_sketch():
    img = blank(size=256)
    plus(img, 128, 128)
    r = post(img, {"crossings": [pt(128, 128)], "terminals": [rec((120, 128), (128, 120), True)]})
    assert r.status_code == 200, r.text


def test_rejects_non_grayscale_png():
    img = blank()
    plus(img, 32, 32)
    r = client.post(
        "/junctions",
        files={"image": ("s.png", png_bytes_rgb(img), "image/png")},
        data={
            "spec": json.dumps(
                {"crossings": [pt(32, 32)], "terminals": [rec((20, 32), (32, 20), True)]}
            )
        },
    )
    assert r.status_code == 400
    assert "颜色类型" in r.json()["detail"]


def test_rejects_non_png_upload():
    r = client.post(
        "/junctions",
        files={"image": ("s.png", b"not png", "image/png")},
        data={"spec": json.dumps({"crossings": [pt(1, 1)],
                                  "terminals": [rec((0, 0), (1, 0), True)]})},
    )
    assert r.status_code == 400


def test_rejects_wrong_content_type():
    img = blank()
    plus(img, 32, 32)
    r = post(
        img,
        {"crossings": [pt(32, 32)], "terminals": [rec((20, 32), (32, 20), True)]},
        content_type="image/jpeg",
    )
    assert r.status_code == 400
    assert "image/png" in r.json()["detail"]


def test_rejects_bad_json():
    img = blank()
    plus(img, 32, 32)
    r = client.post(
        "/junctions",
        files={"image": ("s.png", png_bytes(img), "image/png")},
        data={"spec": "{oops"},
    )
    assert r.status_code == 400


def test_rejects_empty_terminals():
    img = blank()
    plus(img, 32, 32)
    r = post(img, {"crossings": [pt(32, 32)], "terminals": []})
    assert r.status_code == 422


def test_rejects_too_many_crossings():
    img = blank(size=256)
    points = []
    for i in range(13):
        x = 10 + (i % 4) * 60
        y = 10 + (i // 4) * 60
        plus(img, x, y, reach=8)
        points.append(pt(x, y))
    spec = {"crossings": points, "terminals": [rec((10, 10), (18, 10), True)]}
    assert post(img, spec).status_code == 422


def test_rejects_duplicate_crossings_in_spec():
    img = blank()
    plus(img, 32, 32)
    r = post(
        img,
        {"crossings": [pt(32, 32), pt(32, 32)], "terminals": [rec((20, 32), (32, 20), True)]},
    )
    assert r.status_code == 422


def test_rejects_coordinate_at_256():
    img = blank()
    plus(img, 32, 32)
    r = post(img, {"crossings": [pt(256, 32)], "terminals": [rec((20, 32), (32, 20), True)]})
    assert r.status_code == 422


def test_rejects_crossing_outside_actual_image():
    img = blank(size=40)
    plus(img, 20, 20)
    r = post(img, {"crossings": [pt(45, 32)], "terminals": [rec((12, 20), (20, 12), True)]})
    assert r.status_code == 400
    assert "超出图像范围" in r.json()["detail"]


def test_rejects_terminal_on_paper():
    img = blank()
    plus(img, 32, 32)
    r = post(img, {"crossings": [pt(32, 32)], "terminals": [rec((5, 5), (32, 20), True)]})
    assert r.status_code == 400
    assert "不是墨迹" in r.json()["detail"]


def test_rejects_crossing_that_is_not_four_way():
    img = blank()
    hline(img, 32, 10, 54)
    r = post(img, {"crossings": [pt(32, 32)], "terminals": [rec((20, 32), (44, 32), True)]})
    assert r.status_code == 400
    assert "四向交叉点" in r.json()["detail"]


def test_rejects_coincident_terminal_dots():
    img = blank()
    plus(img, 32, 32)
    r = post(img, {"crossings": [pt(32, 32)], "terminals": [rec((20, 32), (20, 32), True)]})
    assert r.status_code == 422


def test_diagonal_stub_conducts_as_ordinary_ink():
    # Ink at (31,31) is 4-adjacent to both the W arm (31,32) and the N arm
    # (32,31), so it bridges them whatever the crossing does.  The W-N record is
    # therefore satisfied by both states and the crossing cannot be decided.
    img = blank()
    plus(img, 32, 32)
    img[31, 31] = 0
    r = post(img, {"crossings": [pt(32, 32)], "terminals": [rec((20, 32), (32, 20), True)]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["crossings"][0]["state"] == "uncertain"
    assert body["consistent_assignments"] == 2


def test_contradictory_records_conflict():
    # The same two dots cannot be both verified connected and verified open.
    img = blank()
    plus(img, 32, 32)
    r = post(
        img,
        {
            "crossings": [pt(32, 32)],
            "terminals": [
                rec((20, 32), (32, 20), True),
                rec((20, 32), (32, 20), False),
            ],
        },
    )
    assert r.status_code == 200
    assert r.json()["status"] == "conflict"


def test_duplicate_record_is_harmless():
    img = blank()
    plus(img, 32, 32)
    r = post(
        img,
        {
            "crossings": [pt(32, 32)],
            "terminals": [
                rec((20, 32), (32, 20), True),
                rec((20, 32), (32, 20), True),
            ],
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "resolved"
    assert body["crossings"][0]["state"] == "junction"


def test_rejects_unknown_fields():
    img = blank()
    plus(img, 32, 32)
    r = post(
        img,
        {
            "crossings": [pt(32, 32)],
            "terminals": [rec((20, 32), (32, 20), True)],
            "threshold": 100,
        },
    )
    assert r.status_code == 422
