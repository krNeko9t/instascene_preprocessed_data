# 仓库代码 Map

## 顶层目录

```
instascene_preprocessed_data/
├── scripts/                  # 全部代码（入口脚本 + instascene 包）
├── prompts/                  # VLM prompt 模板（.prompt，Python format 字符串）
├── configs/                  # view compose 拼图 spec（JSON）
├── assets/                   # 物理感知本体 ontology v1（JSON schema + 模板）
├── docs/                     # ← 本文档
├── outputs/                  # 每次运行的结果（run_<ts>/<dataset>/<scene>/）
│
├── 3dovs/  lerf/  zipnerf/   # 本地预处理场景数据（+ 同名 .zip 原始包）
│   └── <scene>/
│       ├── images/           #   RGB（00.jpg ...）
│       ├── sam/mask/         #   PNG 实例 mask（像素值=实例ID，跨视角一致）
│       ├── id_maps/          #   NPY 实例 id map（2D int）
│       └── sparse/0/         #   COLMAP 稀疏重建
│
├── visualize_sam_masks.py    # 独立工具：mask 上色可视化
├── .vscode/launch.json       # 现成调试配置（含常用参数组合）
└── .cursor/plans/            # 当初的 pipeline 设计计划文档
```

## scripts/ 代码地图

```
scripts/
├── vlm_gen_scene_desc.py        [209行] ★主入口：VLM 物体标注 pipeline CLI
├── sample_scenes.py             [ 55行] 入口：InsScene-15K 场景抽样 → manifest
├── batch_manifest_stats.py      [142行] 入口：manifest 批量统计视角/物体数
├── ask_vlm_image_audit.py       [ 83行] 入口：VLM 多图可见性探针
│
└── instascene/
    ├── types.py                 [ 36行] Literal 类型；支持的图片后缀；overlay style 校验（唯一实现）
    ├── imaging.py               [ 40行] 共享图像 IO：读图转 RGB、JPEG/base64 编码（唯一实现）
    │
    ├── vlm/                     ★ 标注 pipeline 核心
    │   ├── pipeline.py          [271行] process_scene 主编排：并发、断点续传、落盘、debug dump
    │   ├── config.py            [153行] PipelineConfig 及 5 个子配置；默认 prompt/compose spec
    │   ├── compose.py           [187行] compose spec 类型化模型（ComposeSpec 等）+ 解析/校验
    │   ├── selector.py          [ 87行] 视角质量筛选：阈值过滤 + quality_score 排序取 top-K
    │   ├── preparer.py          [241行] 面板渲染（origin/mask/highlight/overlay/crop）+ 拼图合成
    │   └── client.py            [123行] AsyncOpenAI 封装（重试/限流）；audit_images 同步探针
    │
    ├── scene/                   场景数据加载
    │   ├── models.py            [ 42行] ViewRecord / SceneData / ScenePathRecord
    │   ├── pairing.py           [ 83行] 图像↔mask 配对（stem / infinigen 两种策略）
    │   ├── loaders/
    │   │   ├── scene.py         [174行] load_scene：npy|png|sam2_json 三来源统一成 SceneData
    │   │   ├── masks.py         [ 22行] 单文件读取：png灰度 / npy 2D
    │   │   └── sam2.py          [ 82行] auto_masks.json RLE 解码（pycocotools）→ id map
    │   └── discovery/
    │       ├── instascene.py    [ 28行] images + id_maps/sam/mask 布局发现
    │       └── ins_scene_15k.py [ 64行] InsScene-15K 三类目录布局扫描
    │
    ├── manifest/                场景抽样清单（辅链）
    │   ├── sample.py            [224行] 通用抽样 CLI 骨架 + manifest 写出
    │   ├── sample_registry.py   [ 65行] 三数据集注册表（新增数据集改这里）
    │   ├── io.py                [ 84行] manifest 加载/校验/路径解析
    │   ├── models.py            [ 51行] manifest 条目 / 解析后路径 dataclass
    │   └── roots.py             [ 24行] InsScene-15K 共享存储写死路径
    │
    └── stats/
        └── scene_views.py       [101行] 单场景 n_views/n_objects 统计（供 batch_manifest_stats）
```

## 依赖方向（谁 import 谁）

```
入口脚本 ──▶ instascene.vlm ──▶ instascene.scene ──▶ instascene.types
         ──▶ instascene.manifest ──▶ instascene.scene.{models,discovery}
         ──▶ instascene.stats ──▶ instascene.{manifest,scene}

vlm 内部:  pipeline ─▶ {config, selector, preparer, client}；preparer ─▶ {compose, imaging}
scene 内部: loaders/scene ─▶ {loaders/masks, loaders/sam2, pairing, models}
```

单向依赖、无环；`types.py` 是最底层公共层。

## 数据流文件格式速查

| 文件 | 产自 | 结构要点 |
|---|---|---|
| manifest JSON | `sample_scenes.py` | `{dataset_roots, scenes:[{dataset_id, scene_id}], sampling, ...}` |
| 统计 JSON | `batch_manifest_stats.py` | `{scenes:[{dataset_id, scene_id, n_views, n_objects, error?}], totals:{...}}` |
| 标注结果 JSON | `vlm_gen_scene_desc.py` | 数组，每条 `{id, instance_id, n_views, mode, response(解析后的dict或原文), parse_failed?}` |
| `debug_inputs/object_XXX/` | 同上 | `batch_00_prompt.txt` + `batch_00_meta.json` + 实际发送的 jpg |
| `auto_masks.json` | SAM2（外部） | `{masklet: [[RLE,...] 每帧], ...}`，RLE 序号 k → 实例 id k+1 |
| `assets/*.json` | 手工维护 | 本体字段枚举 + 标注规则；`instance_record.template.json` 为单实例记录模板 |

## 定位问题的入口

| 想改/查什么 | 去哪 |
|---|---|
| 加/改 CLI 参数 | `vlm_gen_scene_desc.py` + `vlm/config.py` |
| 标注质量差、视角选得不对 | `vlm/selector.py` 阈值与 quality_score |
| 换发给 VLM 的图片形式 | `configs/*.json` 写新 spec；新面板变体：`vlm/compose.py` 的 `KNOWN_PANEL_VARIANTS` + `vlm/preparer.py:_render_variant` |
| 换 prompt / 标注 schema | `prompts/` 新建 .prompt；枚举定义对照 `assets/` |
| 支持新 mask 格式 | `scene/loaders/scene.py:load_scene` 加分支 |
| 支持新的远端数据集抽样 | `scene/discovery/ins_scene_15k.py` 写 discover + `manifest/sample_registry.py` 注册 |
| 模型收到几张图？ | 先看 `outputs/.../debug_inputs/`，再用 `ask_vlm_image_audit.py` 探针 |
| 请求失败 | `outputs/.../errors.log`；重试参数在 `vlm/config.py:VlmConfig` |
