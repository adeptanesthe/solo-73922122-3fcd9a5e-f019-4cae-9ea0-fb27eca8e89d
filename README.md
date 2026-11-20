# 图纸分块定位 + 接线交叉点核查 API

同一服务提供两条互相独立的判定流程：

1. **`POST /align`**：把大幅图纸切块扫描得到的多张图块，按已声明的重叠图对与整数平移搜索窗，
   求出各块在整图上的位置（下文“分块定位”）。
2. **`POST /junctions`**：对已经细化为单像素横竖线的黑白线稿，判定每个四向交叉点是**接点**、
   **跨线**还是**仍需核对**（见[接点核查](#接点核查post-junctions)）。

技术栈：FastAPI + NumPy（Pillow 仅用于 PNG 解码）。提供 Docker Compose 与可交互的
`/docs`。

## 快速启动

```bash
docker compose up --build
# 打开 http://localhost:8000/docs
```

生成一套演示图与请求：

```bash
python scripts/make_demo.py demo          # 需要先 pip install -r requirements-dev.txt
python scripts/make_junction_demo.py demo/junctions
docker compose up --build                 # 另一个终端
bash demo/request.sh                      # 期望 status = "aligned"
bash demo/junctions/request.sh            # 期望 status = "needs_review"
```

不用 Docker 时：

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
python scripts/demo_client.py --base-url http://localhost:8000
python scripts/junction_demo_client.py --base-url http://localhost:8000
pytest
```

## 接口

`POST /align`，`multipart/form-data`：

| 字段 | 说明 |
| --- | --- |
| `images` | 3～5 张 **8 位灰度 PNG**（PNG 颜色类型 0、非隔行、每边 ≤ 1024 px）。上传顺序即图号 `0..n-1`，图 0 为基准。 |
| `spec` | JSON 字符串，结构见下。 |

`GET /health`、`GET /docs`（Swagger UI）、`GET /redoc`、`GET /openapi.json`。

### spec 结构

```jsonc
{
  "pairs": [
    { "a": 0, "b": 1,
      "window": { "dx_min": 446, "dx_max": 454, "dy_min": -2, "dy_max": 2 } },
    { "a": 1, "b": 2, "candidates": [[-450, 450], [-451, 450]] }
  ],
  "min_overlap_pixels": 20000,
  "max_mean_abs_diff": 3.0
}
```

* `pairs`：声明哪些图对重叠。每对三选一之外的两种写法**必须且只能**给一个：
  * `window`：闭区间矩形，`(dx_max-dx_min+1)*(dy_max-dy_min+1) ≤ 81`；
  * `candidates`：至多 81 个显式整数平移，自动去重。
* `min_overlap_pixels`：整组共用的最少重叠像素数。
* `max_mean_abs_diff`：整组共用的平均绝对灰度差（MAD）上限，0～255。

### 坐标约定

图对 `(a, b)` 的平移 `(dx, dy)` 表示 **b 的左上角位于 a 的坐标 `(dx, dy)`** 处：

```
(x_b, y_b) = (x_a + dx, y_a + dy)
```

返回的 `positions[i]` 是第 i 张图左上角在整图坐标中的位置，图 0 恒为 `(0, 0)`。

### 判定逻辑

1. **连通性**：把声明的图对当作无向边，若某图块无法从图 0 到达，立即返回
   `disconnected`，并在 `rescan_image_id` 指出建议补扫的一块（取连通度最低者，平局取小编号）。
2. **图对达标**：对每对枚举搜索窗内的每个整数平移，计算重叠矩形内的 MAD。一个平移被接受需同时满足
   *重叠像素数 ≥ `min_overlap_pixels`* 且 *MAD ≤ `max_mean_abs_diff`*。
   任一对没有任何被接受的平移 → `no_solution`，`failing_pairs` 给出原因
   （`insufficient_overlap` 或 `mean_abs_diff_too_high`）与最佳尝试值。
3. **全局布局**：每对可能有多于一个达标平移。求解器在图对图上建生成树（优先候选少的边），
   枚举树边候选组合，剪掉任何与已放置节点冲突的组合，并在叶子处校验其余冗余图对；
   不同的位置向量即不同的布局。

### 响应

HTTP 状态码只表示请求是否合法；对齐结论在 `status` 字段（均返回 200）：

| `status` | 含义 | 关键字段 |
| --- | --- | --- |
| `aligned` | 所有图对同时达标且布局唯一 | `positions`、`pairs`、`mosaic_bounds`、`layout_unique=true` |
| `disconnected` | 有图块连不到首图 | `unreachable_image_ids`、`rescan_image_id` |
| `no_solution` | 无法同时对齐 | `failing_pairs`（单对不达标）或空（各对单独达标但互相矛盾） |
| `ambiguous` | 多解，位置不唯一 | `uncertain_image_ids`、`alternative_layouts`、`search_truncated` |

`pair_candidate_counts` 给出每个声明图对的达标平移个数，便于诊断阈值是否过松。

错误返回：`400` 图像非法（格式/位深/颜色类型/尺寸/数量/`Content-Type`）或 `spec` 不是 JSON；
`422` `spec` 结构不合法、图号越界、图对重复。

## 接点核查（`POST /junctions`）

旧接线图褪色后，交叉点到底是“两线相接（接点）”还是“横竖跨线”往往看不清。这个接口用
**端子对通断记录**把整张线的接法约束出来，逐点告诉档案员哪里必须查原件。

`multipart/form-data`：

| 字段 | 说明 |
| --- | --- |
| `image` | 1 张 **8 位灰度 PNG**（颜色类型 0、非隔行、每边 ≤ 256 px），已细化为单像素横竖线。 |
| `spec` | JSON 字符串，结构见下。 |

```jsonc
{
  "crossings": [{"x": 24, "y": 40}, {"x": 56, "y": 40}],   // 至多 12 个四向交叉点
  "terminals": [
    {"a": {"x": 8, "y": 40}, "b": {"x": 72, "y": 40}, "connected": true}
  ]                                                        // 每个点两个普通墨点 + 已核实通断
}
```

### 判定规则

* **墨迹与连通**：灰度 < 128 视为墨迹；普通墨点按**四邻接**连通，未标记的交叉点按普通墨迹处理
  （四臂自然相接）。
* **标记交叉点的两种接法**：
  * `junction`（接点）：四臂全部相接；
  * `crossover`（跨线）：横臂彼此直通、竖臂彼此直通，横竖在交叉处**不接**。
* **整图判定**：一条通断路径可以绕经其他交叉点，因此某点的结论可能取决于别处的接法。
  实现上先把标记交叉点从墨迹中移除，得到若干 4 连通线段，再把每个交叉点的四臂映射到这些线段；
  对一组接法，用并查集按 `junction`（四臂合并）或 `crossover`（W-E、N-S 分别合并）合并后，
  检查全部端子记录是否成立。
* **逐点结论**：枚举至多 `2**12 = 4096` 种接法，只保留满足全部通断记录的那些；
  对每个交叉点，若所有可行接法中它的状态一致，即判为**必为接点**或**必为跨线**，
  否则为 `uncertain`（仍需核对）。
* **冲突**：没有任何接法能满足全部记录时，`status=conflict`，不给出逐点结论。

### 响应

| `status` | 含义 | 关键字段 |
| --- | --- | --- |
| `resolved` | 每个交叉点都唯一确定 | `crossings[*].state` ∈ `junction`/`crossover` |
| `needs_review` | 至少一个点在不同可行接法中状态不同 | 相应点 `state=uncertain` |
| `conflict` | 图像与通断记录冲突 | `crossings=[]`、`consistent_assignments=0` |

`message` 给出中文摘要，`consistent_assignments` 为满足记录的整图接法数量。
错误返回：`400` 图像非法、坐标越界/不在墨迹上、交叉点缺臂或与另一交叉点正交相邻；
`422` JSON 结构不合法、点数超限、坐标重复或两个端子点重合。

### 输入约定

* 交叉点必须落在实际四向交叉处（东/西/南/北四邻都必须是墨迹），且不能落在另一个标记交叉点上；
  两个标记交叉点不得正交相邻（相邻臂会互为对方像素，超出两态模型）。
* 端子点必须是**普通墨点**：位于墨迹上，且不落在标记交叉点上；两个点不得重合。
* 交叉点与通断记录都从整张图出发判定，记录里两个端子所在的线段可以在别处经过任意多个交叉点。

### 已知限制

* 只处理四向（十字）交叉，且约定输入已细化为**单像素**横竖线；线宽大于 1 像素、斜线或三叉
  交点不在模型内：缺臂的交叉点会被拒绝，加粗的笔画可能把本不相接的臂连成一片而导致误判。
* 不处理端点“悬空线头”之类需要其它证据的语义；所有非标记交叉点一律视为普通相接墨迹。
* 若某个交叉点没有任何通断记录能区分，它会如实报 `uncertain`，不会猜测。

## 实现要点

* **解码**：先读 PNG 的 `IHDR` 校验位深=8、颜色类型=0、非隔行、边长≤1024，再用 Pillow 解码并核对尺寸，
  避免把 16 位或调色板图当作灰度图。
* **打分**：不依赖 SciPy。对每个平移取两图重叠区域的 `int16` 绝对差，`int64` 求和后除以像素数；
  窗口最多 81 个位置、最多 10 个图对，单图 1024×1024 的极端情形实测约 2 秒。
* **求解**：见上，`MAX_LAYOUTS` / `MAX_NODES_EXPANDED` 限制最坏工作量；超限时返回
  `ambiguous` 且 `search_truncated=true`，把所有非基准图列为不确定——宁可说不确定，也不谎报唯一。
  预算在尚未找到任何布局时就耗尽的，同样报 `ambiguous` 而不是 `no_solution`，因为此时无法证明无解。
* **接点核查**：交叉点至多 12 个，状态组合至多 4096 种，逐一用并查集校验全部记录；
  最坏情形（4096 种全成立 × 256 条记录）实测约 1.2 秒。复杂度与图像大小线性相关。

## 目录

```
app/
  main.py       FastAPI 路由、spec 解析与错误映射
  models.py     Pydantic 请求/响应模型与上限常量
  imaging.py    严格 PNG 解码（可配置边长上限）、重叠矩形
  alignment.py  单图对搜索窗打分
  solver.py     全局布局枚举与唯一性判定
  service.py    分块定位编排：连通性 → 打分 → 求解
  junctions.py  接点核查：移除交叉点 → 线段连通分量 → 穷举接法 → 逐点判定
  junction_service.py  接点核查编排
tests/          77 个测试（求解器/接点单测 + HTTP 端到端）
scripts/        make_demo.py 生成分块演示；demo_client.py 直接调用 /align；
                make_junction_demo.py 生成接点演示线稿；junction_demo_client.py 调用 /junctions
Dockerfile      python:3.11-slim、非 root、HEALTHCHECK
docker-compose.yml  端口 8000、只读根文件系统、健康检查
```

## 已知限制

* 只支持整数像素平移，不处理旋转、比例差或亚像素错位；题目要求同比例、同朝向。
* 判定“达标”只看重叠区灰度 MAD；若重叠区是空白或高度重复纹理，可能出现多解，此时按设计返回
  `ambiguous` 而不是猜测。
* MAD 使用全图块的重叠矩形直接比较，不做事先的亮度/对比度归一化；扫描曝光差异需通过
  `max_mean_abs_diff` 放宽，或由扫描端统一曝光。
