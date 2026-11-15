# 适配契约（人类可读速览）

> 机器可读版：`/app/CONTRACT.json`；每任务摘要：`/app/models/<TASK>/meta.json`。
> **预检/冻结适配器只需要读这三处，不需要遍历 /app 源码。**

## 1. 入口只有一个

```
python -u /app/harness/entry.py        # 由 HARNESS_MODE 决定行为
```

| `HARNESS_MODE` | 行为 | 退出码 |
| --- | --- | --- |
| `loop`（默认） | 跑自进化闭环，产物写 `/model/<TASK>/` 与 `/output/model/<TASK>/` | **0**（含兜底路径） |
| `serve` | 官方 predict 协议（stdin JSONL → stdout JSONL） | 0 |
| `selftest` | 自检 + 三包各 1 条真实样本推理 | 0=通过 / 1=有缺失 |

镜像默认 `CMD`（等价于）：

```
--task all --stream train --windows 4 --policy rule --swan off --out /output
```

常用参数：`--task all|A|B|C`、`--stream auto|official|train`、`--windows N`、
`--policy auto|rule|llm`、`--swan off`、`--out /output`、`--previous /previous`、
`--model-out /model`。

> **退出码约定**：闭环无论"跑完"还是"因外部原因没跑成"，只要交付产物已落盘就返回 **0**。
> 若真的失败，`/model/<TASK>/manifest.json` 里的 `failure` 字段会写清原因
> （例：`FileNotFoundError: 找不到 B 包数据`），审计时不会被"退出码 0"掩盖。

## 2. 三个任务与它们的检查点

**交付位就是 `checkpoint.npz`** —— 平台评测读这个文件名，并用它自己的
`Model.predict(原始特征)` 推理（**不会**施加我们 metadata 里的预处理）。
所以三个任务的 `checkpoint.npz` 都是"开箱即用、只吃原始特征"的模型。

| 任务 | 类型 | 特征 | 交付检查点（= 平台读取） | 预处理 | 官方 validation |
| --- | --- | --- | --- | --- | --- |
| A | 分类（6 类） | 128 | `/app/models/A/checkpoint.npz` | 无 | Macro-F1 0.7268 |
| B | 回归（分钟） | 34 | `/app/models/B/checkpoint.npz`（蒸馏版） | 无 | RMSE 536.82 |
| C | 回归（循环） | 24 | `/app/models/C/checkpoint.npz` | 无 | RMSE 51.56 |

内部实验档案（不参与交付，保留以便审计/复现）：

| 文件 | 说明 |
| --- | --- |
| `models/<T>/checkpoint_official_baseline.npz` | 官方起点（A/B/C 基线） |
| `models/B/checkpoint_rank.npz` | 秩变换模型 RMSE 369.47。**依赖外部秩变换**（分位表在 metadata 里），
平台推理路径不施加，直接改名成 `checkpoint.npz` 会让预测被 ReLU 夹到 0。它现在是 B 蒸馏版的**教师** |

> A/C 的 `checkpoint_best.npz` 与 `checkpoint.npz` 是同一份模型，留着只为对照历史。

## 3. 推理

- **推荐**：`HARNESS_MODE=serve python -u /app/harness/entry.py`
  —— 读 stdin 每行 `{"inputs": {...}}`，写 stdout 每行 `{"outputs": ...}`，
  **自动按特征 schema 识别 A/B/C**（三条流共用一个进程也互不污染），
  并自动施加该检查点需要的预处理（B 包的秩变换）。
- **桥接**：`from baselines.predict import predict` → `predict(model, inputs)`，
  与官方 `predict.py` 签名同形；`model` 可以是已加载的 Model、检查点路径，或 `None`（自动识别）。

> 为什么不直接用 `vendor/<TASK>/predict.py`：官方三个包的 `predict.py` 都以顶层模块名
> `model` 导入，同进程混用会拿到前一个任务的缓存；而且它们不做 B 包必需的秩变换
> （直接喂原始特征，B 的预测会从 469 分钟掉到 0）。`/app/baselines/predict.py` 修掉了这两点。

## 4. 训练（续训）与交付产物

- 函数：`/app/vendor/<TASK>/train.py::train_run(profile, train_path, validation_path, output, resume=<父检查点>, epochs, learning_rate, batch_size, stage_id)`
- **父检查点解析顺序**：`--checkpoint` → `/previous`（目录或单文件，多种命名都探测；
  也可用 `--previous` / `HARNESS_PREVIOUS` 改位置）→ 镜像自带
  `models/<TASK>/` 下 `meta.json` 的 `default_checkpoint`。
  父检查点 **加载失败会自动回落** 到内置模型，不中断闭环。
- **产物（两处都写）**：
  - `/model/<TASK>/checkpoint.npz` + `manifest.json`（平台取件路径）
  - `/output/model/<TASK>/checkpoint.npz` + `manifest.json`（兼容取件路径）
- `manifest.json` 关键字段：`origin`（`bundled`=内置模型 / `evolved`=闭环真把挑战者推上线了）、
  `sha256`、`parent_checkpoint`、`parent_origin`、`metric_series`、`promotions`、
  `updates_used`、`epochs_used`、`training_seconds`、`failure`（若有）。
- 逐轮证据：`/output/<task>-<stream>-<时间戳>/windows.jsonl`、`decisions.jsonl`、`report.md`。
- 约束：参数量 < 100,000；续训时 **hidden 宽度与 seed 不可变**。

### 兜底交付（重要）

入口在跑闭环**之前**就先把内置检查点落成交付产物；闭环中途失败也不影响产物已在盘上。
这保证「不低于静态基线」这条硬要求不会因为环境问题（`/data` 没挂、数据为空、超预算、
父检查点 schema 不符）变成 0 分。

## 5. 策略层

- **默认 `rule`**（确定性阈值表），保证无网络也能跑完。
- LLM 可选：存在 `SAMPLER_ENDPOINT` / `PYTRIO_BASE_URL` / `OPENAI_BASE_URL` 等环境变量时启用
  `Qwen/Qwen3.5-4B`（PyTRiO，OpenAI 兼容 `/api/openai/v1`）；
  请求带 `enable_thinking: false` 与 `n: 3`（自洽性投票）；**任何异常自动回落规则版**。
- 动作集合：`hold / recompute_stats / finetune_1ep / finetune_3ep / swap_candidate / rollback`。
- 守门余量（`gate.margin`，挑战者必须超过冠军这个幅度才上线）：
  A=0.005（Macro-F1）、B=5.0（分钟）、C=0.5（循环）。

## 6. 数据与限制

- `HARNESS_DATA` 默认 `/data`；若找不到该任务的 `train.jsonl`，依次回落
  `/app/data/public/<TASK>/`（镜像内公开数据副本，仅用于离线自检/兜底）。
  目录名用 `A`/`B`/`C` 或中文名都能识别。
- 硬上限：`/app/harness/config.json` 的 `caps`（更新 4 次 / 8 epoch / 600 秒 / 12 窗口），触发即停。

## 7. 合规

验证集只测不改（`--stream official` 强制禁用更新）；选型/守门只用 train 内部留出窗口；
`id`/`meta` 不进模型输入；不自造标签（不做 RUL 封顶、不做健康等级分箱）；
预处理分位表只在官方 `train.jsonl` 上拟合并随检查点一起保存。
