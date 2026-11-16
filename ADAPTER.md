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

### 4.0 平台契约入口：`python -m baselines.train`（**最关键的一条**）

按官方《续训指南》，每轮由平台执行：

```sh
python -m baselines.train --profile <A|B|C> \
  --train public/train.jsonl --validation public/validation.jsonl \
  --from-baseline checkpoint.npz --epochs 5 --output round-1 --stage-id round-1
# 后续轮：--resume round-1/checkpoint.npz --output round-2 --stage-id round-2
```

产物目录（必须是新目录）：`checkpoint.npz` / `training.json` / `history.jsonl`。
`training.json` 的 `parent_checkpoint_sha256` 指向真实起点，`initial_steps → final_steps` 递增。

> **这一条曾经缺失**：我们之前只有 `vendor/<TASK>/train.py`（顶层模块，自用），
> `/app/baselines/` 里没有 `train.py`，平台执行上面的命令会失败，
> 于是每轮记录成「未训练，保留父模型」。现在 `/app/baselines/train.py` 已按契约实现，
> 并注入我们验证过的配方（见 `baselines/README.md`）。

### 4.1 备选：程序化调用

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

### 4.2 起点检查点优先级（镜像内固定为 v6 语义）

镜像 `ENV HARNESS_START_POLICY=previous`，即：

```
--checkpoint → /previous（平台上一轮选中的模型）→ 镜像自带 models/<TASK>/
```

**为什么必须是 previous 优先**：平台每轮会把"当前被选中的模型"挂进来，闭环从它继续，
才形成跨轮累计（实测 B 的隐藏集 RMSE 从 720 降到 203.5 就是这个机制）。
若反过来用镜像自带模型每轮重启，会把已累计的收益清零。

### 兜底交付（重要）

入口在跑闭环**之前**就先把内置检查点落成交付产物；闭环中途失败也不影响产物已在盘上。
这保证「不低于静态基线」这条硬要求不会因为环境问题（`/data` 没挂、数据为空、超预算、
父检查点 schema 不符）变成 0 分。

## 5. 策略层

### 5.0 模型访问注入点（平台用 RPC 网关时的忠实接法）

平台可能**不是**给 OpenAI 兼容 HTTP 端点，而是只暴露一个 host gateway / RPC 入口。
为此我们提供了三个**由平台决定接线**的注入点，不替换传输语义、不改我们的调用协议：

| 注入方式 | 环境变量 | 说明 |
| --- | --- | --- |
| **命令式网关**（最通用） | `HARNESS_LLM_CMD` | 指定一个命令：我们把**请求 JSON 写到它的 stdin**，从 **stdout 读回复 JSON**。回复可以是 `{"content":"..."}`、`{"choices":[{"message":{"content":"..."}}]}`、`{"text":...}`、`{"output":...}` 任一形状 |
| **配置外置** | `HARNESS_LLM_CONFIG` | 指向一个 JSON：`{"endpoint"|"base_url"|"model_base_url", "model", "api_key"}` |
| **端点改写** | `HARNESS_MODEL_BASE_URL` / `SAMPLER_BASE_URL` / `SAMPLER_ENDPOINT` / `PYTRIO_BASE_URL` / `OPENAI_BASE_URL` | OpenAI 兼容 HTTP 端点 |

另有 `HARNESS_LLM_PROBE=0` 可跳过启动探测（平台已明确模型可用时省掉一跳）。

**验证方式**（我们已自测通过）：把 `HARNESS_LLM_CMD` 指向任意满足上述 stdin/stdout 约定的命令，
闭环决策来源会变成 `source=llm`，`model_used` 记录该模型，token 用量进 `decisions.jsonl`。

- **镜像默认 `--policy auto`**：上述任一注入存在即启用 LLM；都没有 → 确定性规则版（保证跑完）。
- **模型降级链**：`Qwen/Qwen3.8-27B` → `Qwen/Qwen3.5-4B`；启动探测 16 token / 15 秒。
- 每窗决策记录 `source`（`llm` / `rule` / `rule(fallback)`）与 `model_used`，可自证合规。
- 动作集合：`hold / recompute_stats / finetune_1ep / finetune_3ep / swap_candidate / rollback`。
- **受控停更**：`config.policy.allow_update`（当前 `C=false`，因其更新后 NASA 由 4.26e10 变差到 7.65e10）。
- 守门余量：A=0.005（Macro-F1）、B=5.0（分钟）、C=0.5（循环）。

### 5.0.1 时间预算注入点

平台若给更紧的超时，可用环境变量覆盖配置上限（取两者较小值）：
`HARNESS_MAX_SECONDS` / `HARNESS_MAX_WINDOWS` / `HARNESS_MAX_UPDATES` / `HARNESS_MAX_EPOCHS`。
配合"兜底交付"，即使中途被平台超时打断，`/model/<TASK>/checkpoint.npz` 也已经在盘上。


## 5.1 实验台账（门槛 3 的直接证据）

每个任务的输出目录会写 `experiments.jsonl`，**每窗一行、四段式**：

```json
{"task":"B","round":2,
 "read_history":{"prev_metric":204.56,"ref_metric":204.56,"history_actions":["finetune_1ep"]},
 "hypothesis":"已出现漂移信号：轻量续训（1 epoch、低学习率）可在低成本下缓解",
 "change":"finetune_1ep（epochs=1, 2.1s, steps [825, 1155]）",
 "result":{"metric":190.86,"delta_vs_ref":-13.7,"drift_ks_median":0.488,"by_group":{}},
 "decision":{"action":"rollback","source":"llm","gate":{"adopted":false},"reason":"..."}}
```

字段刻意按赛题门槛"读历史→提假设→做实验→做决策"四段式组织，评委可直接核对。

## 6. 数据与限制

- `HARNESS_DATA` 默认 `/data`；若找不到该任务的 `train.jsonl`，依次回落
  `/app/data/public/<TASK>/`（镜像内公开数据副本，仅用于离线自检/兜底）。
  目录名用 `A`/`B`/`C` 或中文名都能识别。
- 硬上限：`/app/harness/config.json` 的 `caps`（更新 4 次 / 8 epoch / 600 秒 / 12 窗口），触发即停。

## 7. 合规

验证集只测不改（`--stream official` 强制禁用更新）；选型/守门只用 train 内部留出窗口；
`id`/`meta` 不进模型输入；不自造标签（不做 RUL 封顶、不做健康等级分箱）；
预处理分位表只在官方 `train.jsonl` 上拟合并随检查点一起保存。
