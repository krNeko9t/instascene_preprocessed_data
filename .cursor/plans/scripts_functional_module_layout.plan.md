# scripts 按功能聚类重组（替代「仅改名」）

## 上一次方案的问题

把 `vlm`/`tools` 换成 `caption`/`batch` **只是标签**，目录里仍然是同一堆平铺模块；对你关心的「相同功能的代码放在一起」帮助很小。**有区别的作法**是：按 **职责边界** 分子包，让「打开某一目录就只看见一类事」。

## 代码里已经隐含的两条产品与一块共享语义

- **Run 终端**：磁盘场景 → 配对/加载 → 选视角 → 拼图 → 调 VLM → 写 JSON（[`scripts/vlm/run.py`](scripts/vlm/run.py) 链路）。
- **Batch 终端**：manifest → 解析路径 → **用与 run 相同的配对与 id 规则**统计视角数/物体数（[`scene_view_object_stats`](scripts/tools/scene_view_object_stats.py) 刻意依赖 [`data_loader`](scripts/vlm/data_loader.py) / [`view_pairing`](scripts/vlm/view_pairing.py)）。
- **共享内核**：「RGB 与 mask 如何配对、npy/png/sam2_json 如何解码」是两条链路共同的**领域语义**，适合收敛到一个子包，避免 batch 目录再去 import 一个叫 `caption` 或 `vlm` 的含糊大包。

## 目标布局（按功能，不是按「管线 vs 工具」）

### A. `scripts/scene_semantics/`（或 `scene_io/`）— 共享「场景 on-disk 语义」

职责：`Path` / manifest 条目 → 配对、解码 mask、拼出与 [`load_scene`](scripts/vlm/data_loader.py) 一致的视图列表。

建议迁入模块（文件名可先不改，少搅乱 blame）：

- `view_pairing.py`
- `sam2_auto_masks.py`
- `data_loader.py`（若保留文件名，路径变为 `scene_semantics/data_loader.py`）

**batch 侧统计**与 **caption 侧 run** 都只从这里 import 配对与加载；[`scene_view_object_stats`](scripts/tools/scene_view_object_stats.py) 改为 `from scripts.scene_semantics...`，语义上说得通：**统计遵守的就是这套 scene 语义**。

### B. `scripts/caption/` — 只做「多视角 → VLM → 结果」

职责：在 A 提供的 `SceneData` 之上，筛选视角、渲染拼图、调 API、编排断点续跑。

建议结构：

```text
caption/
  config.py          # 管线与 API 参数（保留）
  pipeline.py        # 编排（保留）
  views/
    selector.py      # 由 view_selector 迁入
    preparer.py      # 由 image_preparer 迁入
  api/
    client.py        # vlm_client
  cli/
    run.py           # 主入口
    audit.py         # ask_vlm_image_audit（可选仅移动路径、不改文件名）
```

`pipeline` 的 import 变为：`from scripts.scene_semantics.data_loader import load_scene`、`from caption.views.selector import ...` 等。

### C. `scripts/batch/` — 只做「清单 / 采样 / 聚合统计」

职责：写 manifest、发现路径、采样 job、对 manifest 跑批量统计。

建议结构：

```text
batch/
  manifest/
    schema.py        # infinigen_manifest（迁入）
  paths/
    record.py        # scene_path_record
    discover.py      # scene_path_discover_ins_scene_15k
    roots.py         # ins_scene_15k_roots
  sampling/
    job.py           # scene_path_sample_job
    sample_infinigen_scenes.py / sample_re10k_scenes.py / …（CLI 入口放此处或 batch/cli/）
  stats/
    scene_counts.py  # scene_view_object_stats
    manifest_stats.py # batch_manifest_stats（或保留原名）
```

[`scene_view_object_stats`](scripts/tools/scene_view_object_stats.py) 对共享内核的依赖改为 **`scripts.scene_semantics`**，不再依赖 `caption`，这样 **batch 与 caption 并列**，依赖关系清晰：**二者同依赖底层语义包**。

### D. 根目录 [`scripts/run.py`](scripts/run.py) shim

继续转发到 `caption.cli.run:main`（或 `scripts.caption.cli.run`），对外命令不变。

## 与「仅改名 caption/batch」的本质差别

| 作法 | 效果 |
|------|------|
| 只把 `vlm`→`caption`、`tools`→`batch` | 心智模型略好，**文件仍一锅粥** |
| **抽出 `scene_semantics` + caption/batch 内按 views/api/cli、manifest/paths/sampling/stats 分层** | 同类模块物理相邻；batch 不再「挂靠」caption；共享规则有一处真理 |

## 实施顺序（降低一次性炸裂）

1. 新建 `scripts/scene_semantics/`，迁入 `view_pairing`、`sam2_auto_masks`、`data_loader`，包内相对 import；全仓库替换对旧路径的引用。
2. 将 `caption`（当前 [`scripts/vlm`](scripts/vlm) 可先改名或先迁）拆出 `views/`、`api/`、`cli/`，更新 `pipeline`/`run`。
3. 将 `batch`（当前 [`scripts/tools`](scripts/tools)）拆出 `manifest/`、`paths/`、`sampling/`、`stats/`。
4. `compileall`、`python -m ... --help`、必要时跑一次最小 batch 与最小 caption 路径验收。

## 风险

- **diff 较大**：适合单独分支；优先保证行为不变（纯搬家 + import）。
- **若坚持少动目录层级**：最低限度也应做 **A（scene_semantics）**，否则 batch 侧永远像在「借用」caption 的私有模块，这才是你觉得「跟没分一样」的根源。

## 不涉及

- 不改计划附件以外无关文档；不擅自合并多个 `.py` 成一个（除非后续单独重构 PR）。
