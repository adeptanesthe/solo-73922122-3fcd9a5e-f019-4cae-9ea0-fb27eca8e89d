"""Pydantic request/response models for the tile-alignment API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from .imaging import MAX_SKETCH_SIDE

MIN_IMAGES = 3
MAX_IMAGES = 5
MAX_IMAGE_SIDE = 1024
MAX_WINDOW_POSITIONS = 81
MAX_PAIRS = MAX_IMAGES * (MAX_IMAGES - 1) // 2

# Junction check: a small sketch with a handful of ambiguous crossings.
MAX_CROSSINGS = 12
MAX_TERMINAL_RECORDS = 256


class WindowSpec(BaseModel):
    """Rectangular integer translation search window (inclusive bounds).

    The offset ``(dx, dy)`` is the position of image ``b``'s top-left corner in
    image ``a``'s pixel coordinate frame.  A rectangular window must contain at
    most 81 integer positions in total.
    """

    model_config = ConfigDict(extra="forbid")

    dx_min: StrictInt = Field(ge=-1_000_000, le=1_000_000)
    dx_max: StrictInt = Field(ge=-1_000_000, le=1_000_000)
    dy_min: StrictInt = Field(ge=-1_000_000, le=1_000_000)
    dy_max: StrictInt = Field(ge=-1_000_000, le=1_000_000)

    @model_validator(mode="after")
    def _check_bounds(self) -> "WindowSpec":
        if self.dx_min > self.dx_max or self.dy_min > self.dy_max:
            raise ValueError("搜索窗的 min 不能大于 max")
        count = (self.dx_max - self.dx_min + 1) * (self.dy_max - self.dy_min + 1)
        if count > MAX_WINDOW_POSITIONS:
            raise ValueError(
                f"每个图对的搜索窗至多 {MAX_WINDOW_POSITIONS} 个位置，当前为 {count}"
            )
        return self

    def positions(self) -> list[tuple[int, int]]:
        return [
            (dx, dy)
            for dy in range(self.dy_min, self.dy_max + 1)
            for dx in range(self.dx_min, self.dx_max + 1)
        ]


class PairSpec(BaseModel):
    """One declared overlapping image pair.

    ``a`` and ``b`` are zero-based upload indices.  ``window`` and
    ``candidates`` are mutually exclusive ways to describe the search space.
    """

    model_config = ConfigDict(extra="forbid")

    a: StrictInt = Field(ge=0)
    b: StrictInt = Field(ge=0)
    window: WindowSpec | None = None
    candidates: list[tuple[StrictInt, StrictInt]] | None = None

    @model_validator(mode="after")
    def _check_pair(self) -> "PairSpec":
        if self.a == self.b:
            raise ValueError("图对的两张图不能相同")
        if (self.window is None) == (self.candidates is None):
            raise ValueError("每个图对必须且只能给出 window 或 candidates 之一")
        if self.candidates is not None:
            if not (1 <= len(self.candidates) <= MAX_WINDOW_POSITIONS):
                raise ValueError(
                    f"candidates 数量必须在 1 到 {MAX_WINDOW_POSITIONS} 之间"
                )
        return self

    def offsets(self) -> list[tuple[int, int]]:
        """Return search offsets, de-duplicated, preserving input order."""
        raw = self.candidates if self.candidates is not None else self.window.positions()
        seen: set[tuple[int, int]] = set()
        result: list[tuple[int, int]] = []
        for dx, dy in raw:
            key = (int(dx), int(dy))
            if key not in seen:
                seen.add(key)
                result.append(key)
        return result


class AlignSpec(BaseModel):
    """Shared alignment request metadata (sent as a JSON string in the form)."""

    model_config = ConfigDict(extra="forbid")

    pairs: list[PairSpec] = Field(default_factory=list, max_length=MAX_PAIRS)
    min_overlap_pixels: StrictInt = Field(ge=1, le=MAX_IMAGE_SIDE * MAX_IMAGE_SIDE)
    max_mean_abs_diff: float = Field(ge=0, le=255, allow_inf_nan=False)

    @model_validator(mode="after")
    def _check_pairs(self) -> "AlignSpec":
        seen: set[tuple[int, int]] = set()
        for pair in self.pairs:
            key = (min(pair.a, pair.b), max(pair.a, pair.b))
            if key in seen:
                raise ValueError(f"图对 {key} 被重复声明")
            seen.add(key)
        return self


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


class ImageInfo(BaseModel):
    image_id: int
    width: int
    height: int


class ImagePosition(BaseModel):
    image_id: int
    x: int
    y: int


class PairResult(BaseModel):
    pair_id: int
    a: int
    b: int
    dx: int
    dy: int
    overlap_pixels: int = Field(description="重叠像素数（达标时必 ≥ min_overlap_pixels）")
    mean_abs_diff: float = Field(description="重叠区平均绝对灰度差（达标时 ≤ max_mean_abs_diff）")


class PairFailure(BaseModel):
    pair_id: int
    a: int
    b: int
    reason: Literal["insufficient_overlap", "mean_abs_diff_too_high"]
    max_overlap_pixels: int
    best_mean_abs_diff: float | None = None


class MosaicBounds(BaseModel):
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width: int
    height: int


class AlignResponse(BaseModel):
    status: Literal["aligned", "disconnected", "no_solution", "ambiguous"]
    layout_unique: bool
    message: str
    images: list[ImageInfo]
    positions: list[ImagePosition] | None = Field(
        default=None,
        description="仅 status=aligned 时给出；按图号排列，坐标为图左上角，单位像素",
    )
    pairs: list[PairResult] | None = None
    mosaic_bounds: MosaicBounds | None = Field(
        default=None, description="拼接后所有图块的外接矩形（以首图左上角为原点）"
    )
    pair_candidate_counts: list[int] | None = Field(
        default=None, description="每个声明图对在搜索窗内达标的平移个数"
    )
    failing_pairs: list[PairFailure] | None = Field(
        default=None, description="status=no_solution 时给出未达标的图对"
    )
    unreachable_image_ids: list[int] | None = None
    rescan_image_id: int | None = Field(
        default=None, description="status=disconnected 时建议补扫的一块（上传序号）"
    )
    uncertain_image_ids: list[int] | None = Field(
        default=None, description="status=ambiguous 时位置不确定的图块"
    )
    alternative_layouts: list[list[ImagePosition]] | None = Field(
        default=None, description="status=ambiguous 时给出的两组（或多组）可行位置方案"
    )
    search_truncated: bool = Field(
        default=False, description="候选组合数达到搜索上限时为 true，唯一性未获证明"
    )


# ---------------------------------------------------------------------------
# Junction check
# ---------------------------------------------------------------------------


class PointModel(BaseModel):
    """An integer pixel coordinate ``(x, y)``; ``0 <= x < 256`` and same for y."""

    model_config = ConfigDict(extra="forbid")

    x: StrictInt = Field(ge=0, lt=MAX_SKETCH_SIDE)
    y: StrictInt = Field(ge=0, lt=MAX_SKETCH_SIDE)


class TerminalRecordModel(BaseModel):
    """Two ordinary ink dots and the already-verified continuity between them."""

    model_config = ConfigDict(extra="forbid")

    a: PointModel
    b: PointModel
    connected: bool = Field(
        description="两条普通墨点是否导通；true 表示经任意路径连通，false 表示不导通"
    )

    @model_validator(mode="after")
    def _check_distinct(self) -> "TerminalRecordModel":
        if (self.a.x, self.a.y) == (self.b.x, self.b.y):
            raise ValueError("同一条通断记录的两个墨点不能重合")
        return self


class JunctionSpec(BaseModel):
    """JSON body for ``POST /junctions``."""

    model_config = ConfigDict(extra="forbid")

    crossings: list[PointModel] = Field(
        default_factory=list,
        max_length=MAX_CROSSINGS,
        description="待判定的四向交叉点像素坐标，最多 12 个；为空时只校验通断记录与图是否一致",
    )
    terminals: list[TerminalRecordModel] = Field(
        min_length=1,
        max_length=MAX_TERMINAL_RECORDS,
        description="端子对通断记录：每条给出两个普通墨点的坐标与已核实的通断",
    )

    @model_validator(mode="after")
    def _check_crossings_distinct(self) -> "JunctionSpec":
        seen: set[tuple[int, int]] = set()
        for idx, point in enumerate(self.crossings):
            key = (point.x, point.y)
            if key in seen:
                raise ValueError(f"交叉点 {idx} 的坐标 {key} 与其他交叉点重复")
            seen.add(key)
        return self


class CrossingVerdictModel(BaseModel):
    x: int
    y: int
    state: Literal["junction", "crossover", "uncertain"] = Field(
        description=(
            "junction = 必为接点（四臂全部相接）；"
            "crossover = 必为跨线（横竖各自直通，交叉处不接）；"
            "uncertain = 仅凭通断记录仍不能确定，需核对原件"
        )
    )


class JunctionResponse(BaseModel):
    status: Literal["resolved", "needs_review", "conflict"]
    message: str
    consistent_assignments: int = Field(
        description="满足全部通断记录的整图接法数量（交叉点两两状态组合）"
    )
    crossings: list[CrossingVerdictModel] = Field(
        default_factory=list,
        description="status=conflict 时为空；否则与请求中的 crossings 一一对应、同序",
    )
