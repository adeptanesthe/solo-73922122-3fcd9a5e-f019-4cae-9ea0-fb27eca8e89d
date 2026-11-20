"""FastAPI application for locating scan tiles of a large-format drawing."""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from . import __version__
from .imaging import MAX_SKETCH_SIDE, ImageDecodeError, decode_grayscale_png
from .junction_service import JunctionInputError, check_junctions
from .models import (
    MAX_IMAGES,
    MAX_PAIRS,
    MIN_IMAGES,
    AlignResponse,
    AlignSpec,
    JunctionResponse,
    JunctionSpec,
)
from .service import align

DESCRIPTION = """
把一张大幅图纸切块扫描后，用**已声明的重叠图对**与各自的整数平移搜索窗，
确定各块在整张图上的位置。

### 输入
* `images`：3～5 张 **8 位灰度 PNG**（颜色类型 0、非隔行、每边 ≤ 1024 像素）。
  上传顺序即图号 0..n-1，图 0 为定位基准。
* `spec`：JSON 字符串，见下方 schema。包含
  - `pairs`：哪些图对重叠。每对给出 `a`、`b` 以及 `window`
    （`dx_min/dx_max/dy_min/dy_max`，闭区间，至多 81 个整数位置）或
    `candidates`（至多 81 个显式整数平移）；
  - `min_overlap_pixels`：整组共用的最少重叠像素数；
  - `max_mean_abs_diff`：整组共用的平均绝对灰度差上限。

### 坐标约定
第 i 张图左上角在图面上的坐标为 `(x_i, y_i)`；图对 `(a, b)` 的平移 `(dx, dy)`
表示 **b 的左上角位于 a 的坐标 `(dx, dy)`** 处，即
`(x_b, y_b) = (x_a + dx, y_a + dy)`。返回坐标以首图左上角为 `(0, 0)`。

### 结果
* `aligned`：所有图对同时达标且相对布局唯一，返回各图位置与拼接外框。
* `disconnected`：有图块无法经指定图对连到首图；`rescan_image_id` 指出建议补扫的一块。
* `no_solution`：无法同时对齐；`failing_pairs` 给出未达标图对及原因。
* `ambiguous`：多组位置方案都达标；`uncertain_image_ids` 指出位置不确定的图，
  `alternative_layouts` 给出至少两组示例。

### 示例 spec
```json
{
  "pairs": [
    {"a": 0, "b": 1, "window": {"dx_min": 446, "dx_max": 454, "dy_min": -2, "dy_max": 2}},
    {"a": 1, "b": 2, "window": {"dx_min": -454, "dx_max": -446, "dy_min": 448, "dy_max": 452}}
  ],
  "min_overlap_pixels": 20000,
  "max_mean_abs_diff": 3.0
}
```
在上方 `/docs` 页面点击 **Try it out**，先添加 3 张图，再把该 JSON（或 `scripts/make_demo.py`
生成的 `demo/spec.json`）粘贴到 `spec` 字段即可试用。窗口面积两类写法都受 81 个位置的上限约束：
矩形窗口按 `(dx_max-dx_min+1)*(dy_max-dy_min+1) ≤ 81` 计，`candidates` 按条数计。

## 接点核查（`POST /junctions`）

另一条独立的判定流程：给一张已细化为**单像素横竖线**的黑白图（8 位灰度 PNG，每边 ≤ 256 像素）、
最多 12 个四向交叉点坐标，以及若干端子对通断记录，逐点回答该交叉点是**必为接点**、
**必为跨线**还是**仍需核对**。

* 普通墨点按四邻接连通；未标记的交叉点按普通墨迹处理（四臂相接）。
* 标记交叉点只有两种接法：**接点**——四臂全部相接；**跨线**——横竖各自直通、交叉处互不相接。
* 判定看整张图：一条路径可以绕经其他交叉点，因此某点的结论可能取决于别处的接法。
* 综合**所有**满足全部通断记录的整图接法逐点判定；没有一致接法时
  `status=conflict`，只提示图像与通断记录冲突。

### 请求

`multipart/form-data`：`image` 为 1 张 8 位灰度 PNG（颜色类型 0、非隔行、每边 ≤ 256 px），
`spec` 为 JSON 字符串：

```json
{
  "crossings": [{"x": 24, "y": 40}, {"x": 56, "y": 40}],
  "terminals": [
    {"a": {"x": 8, "y": 40}, "b": {"x": 72, "y": 40}, "connected": true}
  ]
}
```

点坐标必须落在图内、位于墨迹上；交叉点必须是真正的四向交叉（四臂都有墨迹），
且不得落在标记交叉点上。`terminals` 至少一项；`crossings` 可为空（此时只校验记录与图是否一致）。

### 响应

| `status` | 含义 |
| --- | --- |
| `resolved` | 每个交叉点都唯一确定 |
| `needs_review` | 至少一个交叉点在所有一致接法中取值不同，需核对原件 |
| `conflict` | 没有任何接法能满足全部通断记录；`crossings` 为空 |

`crossings` 与请求同序，`state` 为 `junction` / `crossover` / `uncertain`；
`consistent_assignments` 为满足记录的不同整图接法数（交叉点两两状态组合）。
`/docs` 中可直接上传线稿试用，或运行 `python scripts/make_junction_demo.py demo/junctions`。
"""


app = FastAPI(
    title="图纸分块定位与接点核查 API",
    version=__version__,
    description=DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    contact={"name": "档案数字化"},
)


@app.get("/health", tags=["meta"], summary="存活检查")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


def _parse_spec(spec_text: str, n_images: int) -> AlignSpec:
    try:
        raw = json.loads(spec_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"spec 不是合法 JSON：{exc}") from exc
    try:
        parsed = AlignSpec.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=json.loads(exc.json(include_url=False, include_context=False)),
        ) from exc
    if len(parsed.pairs) > MAX_PAIRS:
        raise HTTPException(status_code=422, detail=f"图对数量至多 {MAX_PAIRS}")
    for pid, pair in enumerate(parsed.pairs):
        if pair.a >= n_images or pair.b >= n_images:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"图对 {pid} 引用了不存在的图号 {max(pair.a, pair.b)}；"
                    f"本次上传了 {n_images} 张图（图号 0..{n_images - 1}）"
                ),
            )
    return parsed


MAX_UPLOAD_BYTES = 16 * 1024 * 1024  # per tile; a 1024x1024 gray PNG is far below this


def _parse_junction_spec(spec_text: str) -> JunctionSpec:
    try:
        raw = json.loads(spec_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"spec 不是合法 JSON：{exc}") from exc
    try:
        return JunctionSpec.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=json.loads(exc.json(include_url=False, include_context=False)),
        ) from exc


def _read_upload(upload: UploadFile, idx: int) -> bytes:
    """Read one upload with a hard size cap so a huge file cannot exhaust RAM."""
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"图 {idx} 超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB 上限",
        )
    return data


@app.post(
    "/align",
    response_model=AlignResponse,
    tags=["alignment"],
    summary="对齐分块扫描图并返回拼接坐标",
    response_description="对齐结果；HTTP 200 时结果的业务状态见 `status` 字段",
    responses={
        400: {"description": "请求格式或图像不符合要求"},
        422: {"description": "spec 结构或图号引用不合法"},
    },
)
def align_endpoint(
    images: Annotated[list[UploadFile], File(description="3～5 张 8 位灰度 PNG，按图号顺序上传")],
    spec: Annotated[str, Form(description="AlignSpec JSON 字符串（见接口说明）")],
) -> AlignResponse:
    if not (MIN_IMAGES <= len(images) <= MAX_IMAGES):
        raise HTTPException(
            status_code=400,
            detail=f"需要 {MIN_IMAGES}～{MAX_IMAGES} 张图，收到 {len(images)} 张",
        )

    parsed = _parse_spec(spec, len(images))

    decoded = []
    for idx, upload in enumerate(images):
        ctype = (upload.content_type or "").split(";")[0].strip().lower()
        if ctype not in ("image/png", "application/octet-stream", ""):
            raise HTTPException(
                status_code=400,
                detail=f"图 {idx} 的 Content-Type 为 {upload.content_type!r}，只接受 image/png",
            )
        data = _read_upload(upload, idx)
        try:
            decoded.append(decode_grayscale_png(idx, data))
        except ImageDecodeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return align(decoded, parsed)


@app.post(
    "/junctions",
    response_model=JunctionResponse,
    tags=["junction check"],
    summary="判定褪色接线图上各四向交叉点是接点、跨线还是仍需核对",
    response_description="判定结果；HTTP 200 时结论见 `status` 字段",
    responses={
        400: {"description": "图像不符合要求，或交叉点/端子坐标不是合法墨点"},
        422: {"description": "JSON 结构不合法或点数超限"},
    },
)
def junctions_endpoint(
    image: Annotated[UploadFile, File(description="1 张 8 位灰度 PNG，每边 ≤ 256 像素")],
    spec: Annotated[str, Form(description="JunctionSpec JSON 字符串（见接口说明）")],
) -> JunctionResponse:
    ctype = (image.content_type or "").split(";")[0].strip().lower()
    if ctype not in ("image/png", "application/octet-stream", ""):
        raise HTTPException(
            status_code=400,
            detail=f"Content-Type 为 {image.content_type!r}，只接受 image/png",
        )
    data = _read_upload(image, 0)
    try:
        # The junction sketch is capped at 256 px per side, unlike alignment tiles.
        decoded = decode_grayscale_png(0, data, max_side=MAX_SKETCH_SIDE)
    except ImageDecodeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    parsed = _parse_junction_spec(spec)
    try:
        return check_junctions(decoded, parsed)
    except JunctionInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
