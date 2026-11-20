"""Whole-image interpretation of ambiguous four-way crossings.

The input sketch has already been thinned to single-pixel horizontal/vertical
lines.  Every ink pixel is connected to its four orthogonal neighbours, *except*
at the marked four-way crossings, whose wiring is exactly what has to be
decided.  A marked crossing admits two states:

``junction`` (接点)
    all four arms are joined at the crossing pixel;
``crossover`` (跨线)
    the horizontal arms stay connected to each other, the vertical arms stay
    connected to each other, but the two lines do not touch at the crossing.

Because the state of one crossing can change which route a wire takes through
*other* crossings, the decision for a terminal pair is a global connectivity
question, not a local one.  The image is therefore reduced to a small graph:

1. remove the marked crossing pixels from the ink mask;
2. label the remaining ink into 4-connected components ("wire segments");
3. each crossing becomes four arm references to those components;
4. for a given state assignment, union the arm components according to the
   state (junction: all four; crossover: W-E and N-S) and test every terminal
   pair.

There are at most 12 crossings, so all 2**12 state assignments can be checked
exhaustively.  A crossing is reported as ``junction``/``crossover`` only when
every assignment that satisfies all terminal records agrees on its state;
otherwise it is ``uncertain``.  If no assignment satisfies the records, the
sketch and the records conflict.

Unmarked ink pixels are never removed from the components, so an unmarked
four-way pixel simply joins all of its arms and is never treated as ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .imaging import INK_THRESHOLD

# Arm order used throughout the module: W, E, N, S.
_ARMS: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


class JunctionInputError(ValueError):
    """The sketch and the supplied points do not form a valid junction check."""


@dataclass(frozen=True)
class TerminalRecord:
    """One verified continuity record between two ordinary ink dots."""

    a: tuple[int, int]
    b: tuple[int, int]
    connected: bool


@dataclass(frozen=True)
class CrossingVerdict:
    x: int
    y: int
    state: str  # "junction" | "crossover" | "uncertain"


@dataclass
class Interpretation:
    status: str  # "resolved" | "needs_review" | "conflict"
    verdicts: list[CrossingVerdict]
    consistent_assignments: int
    message: str


class _DSU:
    __slots__ = ("parent",)

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        parent = self.parent
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _component_labels(ink: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 4-connected ink components; return (labels, count), -1 = no ink."""
    height, width = ink.shape
    labels = np.full((height, width), -1, dtype=np.int32)
    count = 0
    ys, xs = np.nonzero(ink)
    for start in range(len(ys)):
        y0, x0 = int(ys[start]), int(xs[start])
        if labels[y0, x0] != -1:
            continue
        labels[y0, x0] = count
        stack = [(y0, x0)]
        while stack:
            y, x = stack.pop()
            for dy, dx in _ARMS:
                ny, nx = y + dy, x + dx
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and ink[ny, nx]
                    and labels[ny, nx] == -1
                ):
                    labels[ny, nx] = count
                    stack.append((ny, nx))
        count += 1
    return labels, count


def _validate_dot(
    point: tuple[int, int],
    label: str,
    ink: np.ndarray,
    forbidden: set[tuple[int, int]],
) -> None:
    height, width = ink.shape
    x, y = point
    if not (0 <= x < width and 0 <= y < height):
        raise JunctionInputError(f"{label} ({x}, {y}) 超出图像范围 {width}x{height}")
    if not ink[y, x]:
        raise JunctionInputError(f"{label} ({x}, {y}) 处不是墨迹像素")
    if point in forbidden:
        raise JunctionInputError(f"{label} ({x}, {y}) 落在标记交叉点上")


def interpret(
    image: np.ndarray,
    crossings: Sequence[tuple[int, int]],
    records: Sequence[TerminalRecord],
) -> Interpretation:
    """Decide which crossing states are compatible with all terminal records.

    ``image`` is an 8-bit grayscale array; pixels darker than
    :data:`INK_THRESHOLD` are ink.  Raises :class:`JunctionInputError` when the
    points are not valid four-way crossings / ordinary ink dots.
    """
    height, width = image.shape
    ink = image < INK_THRESHOLD

    crossing_list = [(int(x), int(y)) for x, y in crossings]
    crossing_set = set(crossing_list)
    if len(crossing_set) != len(crossing_list):
        raise JunctionInputError("标记交叉点坐标重复")

    # Every marked crossing must be an ordinary ink pixel with all four arms
    # present, and no two marked crossings may be orthogonally adjacent: their
    # arms would be each other's removed pixels, which the two-state model
    # cannot describe.
    for idx, (x, y) in enumerate(crossing_list):
        _validate_dot((x, y), f"交叉点 {idx}", ink, set())
        for dx, dy in _ARMS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < width and 0 <= ny < height) or not ink[ny, nx]:
                raise JunctionInputError(
                    f"交叉点 {idx} ({x}, {y}) 不是四向交叉点："
                    f"方向 ({dx}, {dy}) 缺少墨迹臂"
                )
            if (nx, ny) in crossing_set:
                raise JunctionInputError(
                    f"交叉点 {idx} ({x}, {y}) 与另一标记交叉点正交相邻；"
                    "请核对输入坐标"
                )

    for idx, record in enumerate(records):
        _validate_dot(record.a, f"第 {idx} 条记录的点 a", ink, crossing_set)
        _validate_dot(record.b, f"第 {idx} 条记录的点 b", ink, crossing_set)

    # Components of ordinary ink, with all marked crossing pixels removed.
    ordinary = ink.copy()
    for x, y in crossing_list:
        ordinary[y, x] = False
    labels, _n_components = _component_labels(ordinary)

    arms: list[tuple[int, int, int, int]] = [
        tuple(int(labels[y + dy, x + dx]) for dx, dy in _ARMS)  # type: ignore[misc]
        for x, y in crossing_list
    ]
    endpoints: list[tuple[int, int]] = [
        (int(labels[record.a[1], record.a[0]]), int(labels[record.b[1], record.b[0]]))
        for record in records
    ]

    # Only components touched by an arm or a terminal can matter.  Renumbering
    # keeps the per-assignment DSU tiny (at most 4*12 + 2*records nodes).
    relevant = {c for arm in arms for c in arm}
    for ca, cb in endpoints:
        relevant.add(ca)
        relevant.add(cb)
    index = {comp: i for i, comp in enumerate(sorted(relevant))}
    arm_idx = [tuple(index[c] for c in arm) for arm in arms]
    endpoint_idx = [(index[ca], index[cb]) for ca, cb in endpoints]
    n_nodes = len(index)

    n_crossings = len(crossing_list)
    consistent: list[int] = []
    for mask in range(1 << n_crossings):
        dsu = _DSU(n_nodes)
        for i, (w, e, n, s) in enumerate(arm_idx):
            if (mask >> i) & 1:  # junction: all four arms joined
                dsu.union(w, e)
                dsu.union(w, n)
                dsu.union(w, s)
            else:  # crossover: horizontal through, vertical through, no contact
                dsu.union(w, e)
                dsu.union(n, s)
        if all(
            (dsu.find(ta) == dsu.find(tb)) == record.connected
            for (ta, tb), record in zip(endpoint_idx, records)
        ):
            consistent.append(mask)

    if not consistent:
        return Interpretation(
            status="conflict",
            verdicts=[],
            consistent_assignments=0,
            message=(
                "图像与端子通断记录冲突：没有任何一种交叉点接法"
                "（接点/跨线）能同时满足全部通断记录。"
            ),
        )

    verdicts: list[CrossingVerdict] = []
    for i, (x, y) in enumerate(crossing_list):
        values = {(mask >> i) & 1 for mask in consistent}
        if values == {1}:
            state = "junction"
        elif values == {0}:
            state = "crossover"
        else:
            state = "uncertain"
        verdicts.append(CrossingVerdict(x=x, y=y, state=state))

    junction_n = sum(v.state == "junction" for v in verdicts)
    crossover_n = sum(v.state == "crossover" for v in verdicts)
    uncertain_n = sum(v.state == "uncertain" for v in verdicts)
    if not n_crossings:
        status = "resolved"
        message = "未标记交叉点；图像与全部通断记录一致。"
    elif uncertain_n:
        status = "needs_review"
        message = (
            f"存在 {len(consistent)} 种满足全部通断记录的整图接法；"
            f"{uncertain_n} 处仍需核对（必为接点 {junction_n} 处，"
            f"必为跨线 {crossover_n} 处）。"
        )
    else:
        status = "resolved"
        message = (
            f"全部 {n_crossings} 处交叉点均已确定：必为接点 {junction_n} 处，"
            f"必为跨线 {crossover_n} 处。"
        )
    return Interpretation(
        status=status,
        verdicts=verdicts,
        consistent_assignments=len(consistent),
        message=message,
    )
