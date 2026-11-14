# 智训引擎 · 工业模型自进化系统（Harness）

第二届北京市大学生“人工智能+”创新大赛 · 九、智训引擎——工业制造场景下的模型训练与超参数优化。

本仓库交付一个**能在漂移数据流上自动维持并提升工业模型性能**的闭环系统，以及打包成
`docker save` 归档的 Harness 镜像。

闭环：**漂移监测 → 诊断归因 → 实验优化 → 评估上线 → 回滚**

---

## 一、仓库结构

```
harness/              闭环系统本体（纯 numpy，无第三方依赖）
├─ runner.py          主循环：监测 → 归因 → 决策 → 优化 → 守门 → 回滚
├─ tasks.py           三包数据适配、数据流切分、按组指标、调用官方训练代码
├─ policy.py          决策器：RulePolicy（确定性）+ LLMPolicy（OpenAI 兼容，带自动回落）
├─ metrics.py         Macro-F1 / RMSE / MAE / bias / NASA 非对称分 / KS 漂移量
├─ runlog.py          JSONL 审计日志 + SwanLab（离线自动跳过）
├─ serve.py           官方 predict.py 协议：stdin JSONL → stdout JSONL（自动识别 A/B/C）
├─ entry.py           统一入口：HARNESS_MODE=loop | serve | selftest
└─ config.json        策略阈值 + 四道硬性预算上限
vendor/{A,B,C}/       官方 CPU 基线代码（model/data/train/predict + configs）
models/{A,B,C}/       官方起点检查点（父检查点指纹可审计）
Dockerfile            镜像定义（CPU、离线、受限机制）
build.ps1             宿主机一键构建（需 Docker Desktop + WSL2）
vm-build.ps1 / .sh    备用构建路径（在 Ubuntu 虚拟机里构建，无需 WSL2）
说明.md               运行说明 / 提交步骤 / 合规边界 / 当前状态
```

## 二、数据准备

官方公开数据不在仓库内（体积与分发约定原因）。把主办方数据包里的
`public/train.jsonl`、`public/validation.jsonl` 按如下放置即可（目录名用 A/B/C 或中文名都行）：

```
data/public/A/train.jsonl        data/public/A/validation.jsonl
data/public/B/train.jsonl        data/public/B/validation.jsonl
data/public/C/train.jsonl        data/public/C/validation.jsonl
```

也可以用环境变量指定：`HARNESS_DATA=<数据根目录>`。

## 三、本地运行

```powershell
$env:PYTHONPATH=''
python harness\runner.py --task B --stream official --windows 6      # 只测不改（官方验证集）
python harness\runner.py --task B --stream train    --windows 4      # 可更新（train 内部切分）
python harness\runner.py --task all --stream auto --windows 4 --swan off
```

产物（每次运行一个目录）：

```
runs/<TASK>-<mode>-<时间戳>/windows.jsonl     逐窗口指标、漂移量、动作、守门结果
runs/<TASK>-<mode>-<时间戳>/decisions.jsonl   决策流水（状态快照 + 动作 + 理由 + 来源）
runs/<TASK>-<mode>-<时间戳>/summary.json      汇总：指标序列、上线/回滚次数、训练耗时
runs/<TASK>-<mode>-<时间戳>/report.md         人读的版本
<out>/model/<TASK>/checkpoint.npz             最终工业模型
<out>/model/<TASK>/manifest.json              sha256 / 父检查点 / 指标序列 / 成本
```

## 四、镜像

```powershell
# 路径一（推荐）：Docker Desktop + WSL2
.\build.ps1 -Doctor        # 体检
.\build.ps1                # 构建 → 自检 → 导出 dist\zhixun-harness_dev.tar

# 路径二（无需 WSL2）：已有 Ubuntu 虚拟机
.\vm-build.ps1 -Ip <虚拟机IP>
```

容器接口：

| 环境变量 | 值 | 作用 |
| --- | --- | --- |
| `HARNESS_MODE` | `loop`（默认）/ `serve` / `selftest` | 闭环 / 管道推理 / 自检 |
| `HARNESS_DATA` | `/data` | 数据挂载点；空则回落到镜像内 `/app/data/public` |
| `SWANLAB_MODE` | `disabled` | 封闭评测无网络 |

## 五、合规边界（硬线）

1. 官方 `validation.jsonl` **只用于衡量**：`--stream official` 强制禁用更新动作；它不进梯度、不参与标准化/目标缩放拟合、不参与守门决策。
2. 更新只使用官方 `train.jsonl` 内部切分的数据（`--stream train`）。
3. 守门（gate）用“下一窗”做冠军/挑战者对比，不用官方验证集做模型选择。
4. `id` / `meta` 不进模型输入；`meta` 仅用于分组诊断与训练期样本加权。
5. 自造标签、RUL 封顶、健康等级分箱等均不使用。
6. 每轮输出新目录，不覆盖历史；父检查点 sha256 与累计步数写入 `training.json` 可审计。

## 六、环境

- Python ≥ 3.10，仅需 `numpy==1.26.4`（镜像内已固定）。
- SwanLab 仅在本地开发时使用（`--swan auto`）；镜像内默认关闭，不依赖网络。
- 全部计算为 CPU，三包一轮训练均在秒级到 1 分钟内。
