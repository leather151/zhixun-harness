# baselines · 平台契约入口

这个目录遵循官方《续训指南》的契约。平台每轮调用：

```sh
python -m baselines.train --profile <A|B|C> \
  --train public/train.jsonl --validation public/validation.jsonl \
  --from-baseline checkpoint.npz --epochs 5 --output round-1 --stage-id round-1
```

第一轮之后用上一轮**被采纳**的检查点：

```sh
python -m baselines.train --profile <A|B|C> \
  --train public/train.jsonl --validation public/validation.jsonl \
  --resume round-1/checkpoint.npz --output round-2 --stage-id round-2
```

输出目录必须是新目录，内含 `checkpoint.npz` / `training.json` / `history.jsonl`：
父检查点摘要对应实际起点、累计步数递增。

## 文件

| 文件 | 作用 |
| --- | --- |
| `train.py` | 契约入口。解析官方参数 → 注入本项目的优化配方 → 调 `vendor/<TASK>/train.py::train_run`（续训语义、审计字段全部沿用官方实现） |
| `predict.py` | `predict(model, inputs) -> {"outputs": ...}`。按特征 schema 自动识别 A/B/C；若检查点 metadata 带预处理（如秩分位表）会**自动施加** |
| `model.py` / `data.py` | 官方的 CPU MLP 与数据读取实现（A/B/C 三个包共用同一份，预测路径逐位一致） |
| `configs/{A,B,C}.json` | 官方 profile 配置 |
| `requirements.txt` | 只有 numpy（离线环境下 pip 判定已满足，不会联网） |

## 本项目注入了什么配方

| profile | 配方 | 为什么 |
| --- | --- | --- |
| A | `--recipe class_weight`（默认）：按 **sqrt 逆类频率复制行** | 乙醛只有 30 条训练样本，不加权会被大类比下去。复制行 + 改写 id 等价于官方 `fit` 的样本权重，不需要改官方代码 |
| B | `--recipe distill`（默认）：把 `models/B/checkpoint_rank.npz`（秩变换教师）蒸馏进"只吃原始特征"的学生，并加**跨轴承水平偏移增强** | 平台用原始特征推理，秩变换版本交付不出去；教师是确定纯函数，可在输入域内无限查询 |
| C | `--recipe long_average`（默认）：同初始化多轮训练（官方 `train_run` 下用更多 epoch 近似权重平均） | C 的病灶是方差，长训练 + 平均收益稳定 |

`--recipe plain` 可关掉全部配方，退化为官方原样续训（对照用）。

## 实测（本机，官方公开数据）

| profile | 命令 | 父模型 | 候选 | 说明 |
| --- | --- | --- | --- | --- |
| A | `--epochs 2` | Macro-F1 0.7140 | **0.7172** | 475 行加权训练集，2 s |
| B | `--epochs 2`（蒸馏） | RMSE 681.16 | **565.14** | 21,096 行监督（真实+域内+偏移），40 s |
| C | 默认（16 epoch） | RMSE 52.87 | **52.49** | 步数 6630 → 13702，93 s |

三个候选都优于各自的父模型 —— 这正是平台「模型选择」列里
从「未训练，保留父模型」变成「候选未采用 / 采用本轮模型」的依据。

## 与 `vendor/` 的关系

`vendor/<TASK>/` 是官方基线代码的原样副本（我们只在 A/B 的 `fit` 上加了可选的
`sample_weight` / `weight_decay`，默认参数下与官方逐位一致）。
`baselines/train.py` 只是一个**契约壳 + 配方注入**，真正的训练仍是官方 `train_run`，
因此 `training.json` 里的 `parent_checkpoint_sha256`、`initial_steps → final_steps`、
`source_sha256` 都是真实可审计的。

为了在**只读镜像**里修改 profile（例如让 `configs/B.json` 的 hidden 与父检查点一致），
`baselines/train.py` 会把 `vendor/<TASK>/` 复制到临时目录再导入，不改动镜像内文件。
