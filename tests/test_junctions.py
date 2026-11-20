"""Unit tests for the whole-image junction interpretation."""

from __future__ import annotations

import numpy as np
import pytest

from app.junctions import JunctionInputError, TerminalRecord, interpret
from tests.junction_fixtures import blank, hline, vline


def plus(img: np.ndarray, x: int, y: int, reach: int = 12) -> None:
    hline(img, y, x - reach, x + reach)
    vline(img, x, y - reach, y + reach)


def rec(a, b, connected):
    return TerminalRecord(a=a, b=b, connected=connected)


def verdict_map(result):
    return {(v.x, v.y): v.state for v in result.verdicts}


def test_single_crossing_horizontal_record_cannot_decide():
    img = blank()
    plus(img, 32, 32)
    result = interpret(
        img,
        [(32, 32)],
        [rec((20, 32), (44, 32), True)],
    )
    assert result.status == "needs_review"
    assert result.verdicts[0].state == "uncertain"
    assert result.consistent_assignments == 2


def test_single_crossing_forces_junction_when_turn_conducts():
    img = blank()
    plus(img, 32, 32)
    result = interpret(
        img,
        [(32, 32)],
        [rec((20, 32), (32, 20), True)],
    )
    assert result.status == "resolved"
    assert result.verdicts[0].state == "junction"
    assert result.consistent_assignments == 1


def test_single_crossing_forces_crossover_when_turn_is_open():
    img = blank()
    plus(img, 32, 32)
    result = interpret(
        img,
        [(32, 32)],
        [rec((20, 32), (32, 20), False)],
    )
    assert result.status == "resolved"
    assert result.verdicts[0].state == "crossover"
    assert result.consistent_assignments == 1


def test_single_crossing_conflict():
    # Crossover keeps N-S connected, so N-S=false excludes it; junction keeps
    # W-N connected, so W-N=false excludes junction.  No wiring is consistent.
    img = blank()
    plus(img, 32, 32)
    result = interpret(
        img,
        [(32, 32)],
        [
            rec((32, 20), (32, 44), False),
            rec((20, 32), (32, 20), False),
        ],
    )
    assert result.status == "conflict"
    assert result.verdicts == []
    assert result.consistent_assignments == 0
    assert "冲突" in result.message


def test_record_on_same_component_always_conducts():
    # Two dots on one horizontal line cannot be reported as disconnected.
    img = blank()
    plus(img, 32, 32)
    result = interpret(
        img,
        [(32, 32)],
        [rec((22, 32), (26, 32), False)],
    )
    assert result.status == "conflict"


def build_detour() -> np.ndarray:
    """Two crossings joined by a top detour, plus an independent plus.

        A=(16,32)          B=(48,32)
    P --+                   +-- Q
        |                   |
        +---- y=8 ----------+

    The direct middle of y=32 is cut, so P-Q must turn up at A and down at B:
    the record is satisfiable only when *both* A and B are junctions.
    """
    img = blank()
    hline(img, 32, 4, 17)  # left arm; 17 is A's east arm
    hline(img, 32, 47, 60)  # right arm; 47 is B's west arm
    vline(img, 16, 8, 56)  # A vertical
    vline(img, 48, 8, 56)  # B vertical
    hline(img, 8, 16, 48)  # top detour
    plus(img, 32, 48, reach=8)  # independent crossing C=(32,48)
    return img


def test_route_through_several_crossings_forces_both_junctions():
    img = build_detour()
    result = interpret(
        img,
        [(16, 32), (48, 32), (32, 48)],
        [rec((6, 32), (58, 32), True)],
    )
    assert result.status == "needs_review"
    states = verdict_map(result)
    assert states == {
        (16, 32): "junction",
        (48, 32): "junction",
        (32, 48): "uncertain",
    }
    assert result.consistent_assignments == 2  # the unrelated crossing is free


def test_route_failure_leaves_both_crossings_uncertain():
    img = build_detour()
    result = interpret(
        img,
        [(16, 32), (48, 32)],
        [rec((6, 32), (58, 32), False)],
    )
    assert result.status == "needs_review"
    assert verdict_map(result) == {(16, 32): "uncertain", (48, 32): "uncertain"}
    # crossover/crossover, junction/crossover, crossover/junction all satisfy.
    assert result.consistent_assignments == 3


def test_detour_with_extra_record_becomes_fully_resolved():
    img = build_detour()
    result = interpret(
        img,
        [(16, 32), (48, 32), (32, 48)],
        [
            rec((6, 32), (58, 32), True),
            # At C=(32,48): W=(24,48), N=(32,40).  Only a junction joins them.
            rec((24, 48), (32, 40), True),
        ],
    )
    assert result.status == "resolved"
    assert result.consistent_assignments == 1
    assert verdict_map(result) == {
        (16, 32): "junction",
        (48, 32): "junction",
        (32, 48): "junction",
    }


def test_unmarked_four_way_crossing_is_plain_conductive_ink():
    # An unmarked plus (32,32) joins all arms as ordinary 4-connected ink; a
    # marked plus elsewhere is unrelated and therefore stays uncertain.
    img = blank()
    plus(img, 32, 32)
    plus(img, 12, 50, reach=6)
    result = interpret(
        img,
        [(12, 50)],
        [rec((20, 32), (32, 20), True)],
    )
    assert verdict_map(result) == {(12, 50): "uncertain"}
    assert result.status == "needs_review"


def test_rejects_crossing_without_four_arms():
    img = blank()
    hline(img, 32, 10, 54)  # only horizontal ink, no vertical arms
    with pytest.raises(JunctionInputError, match="不是四向交叉点"):
        interpret(img, [(32, 32)], [rec((20, 32), (44, 32), True)])


def test_rejects_crossing_outside_image():
    img = blank(size=32)
    with pytest.raises(JunctionInputError, match="超出图像范围"):
        interpret(img, [(40, 10)], [rec((5, 5), (6, 5), True)])


def test_rejects_terminal_on_crossing():
    img = blank()
    plus(img, 32, 32)
    with pytest.raises(JunctionInputError, match="标记交叉点"):
        interpret(img, [(32, 32)], [rec((32, 32), (44, 32), True)])


def test_rejects_adjacent_crossings():
    # Two pluses one pixel apart on the same row; each has four arms, but the
    # removed pixels form a two-wide band the model cannot treat as two points.
    img = blank(size=80)
    for x in (20, 21):
        vline(img, x, 10, 60)
    hline(img, 40, 5, 70)
    with pytest.raises(JunctionInputError, match="正交相邻"):
        interpret(
            img,
            [(20, 40), (21, 40)],
            [rec((8, 40), (60, 40), True)],
        )


def test_rejects_duplicate_crossings():
    img = blank()
    plus(img, 32, 32)
    with pytest.raises(JunctionInputError, match="重复"):
        interpret(img, [(32, 32), (32, 32)], [rec((20, 32), (44, 32), True)])


def test_twelve_crossings_are_enumerated():
    # Twelve independent pluses: 2**12 assignments all satisfy one fixed record
    # on the first plus, so that one is forced and the rest stay uncertain.
    img = blank(size=240)
    points = []
    for i in range(12):
        x = 20 + (i % 4) * 60
        y = 20 + (i // 4) * 60
        plus(img, x, y, reach=10)
        points.append((x, y))
    result = interpret(
        img,
        points,
        [rec((points[0][0] - 5, points[0][1]), (points[0][0], points[0][1] - 5), True)],
    )
    assert result.consistent_assignments == 2**11
    assert result.verdicts[0].state == "junction"
    assert all(v.state == "uncertain" for v in result.verdicts[1:])
