# Sampler 微调记录（决策模型 · 可复现）

> 这一页回答表单里的「**完整 Sampler 权重路径**」。

## 0. 直接填这个

| 字段 | 值 |
| --- | --- |
| 完整 Sampler 权重路径 | `trio://run_4e2okzovpr66/sampler_weights/zhixun-policy-v2` |
| Sampler 权重路径（若要短版） | `run_4e2okzovpr66/sampler_weights/zhixun-policy-v2` |
| 模型类型 | **「持久化 Sampler 微调权重」**（此前以为只能选基础模型，是因为没试过训练接口——见 §6） |
| 基座模型 | `Qwen/Qwen3.5-4B`（PyTRiO） |
| LoRA rank | 32（train_attn / train_mlp / train_unembed 全开） |

**同账号上还有一个 v1**：`trio://run_hibf60zaajlz/sampler_weights/zhixun-policy-v1`。
v1 的损失解析有 bug（日志里是 nan），v2 是同一份数据、把 loss 字段取对之后重训的，
**交 v2**。两个权重在 PyTRiO 控制台都能看到（类型 `sampler`，各 357,985,385 字节）。

## 1. 这个"决策模型"是干什么的

闭环每一轮要做一次决策（继续观察 / 重算统计 / 轻量续训 / 激进续训 / 回滚）。
规则版是确定性阈值表；**LLM 版**把一个窗口的监测+归因结果读成 JSON，输出：
`{"action": "...", "reason": "..."}`。

这一版微调就是把这个决策**行为**克隆到 `Qwen3.5-4B` 上——不是灌知识，是让它学会
"看到什么样的漂移/指标变化该做什么动作"。

## 2. 训练语料：全部来自我们自己的真实闭环轨迹

生成脚本：`harness/build_policy_sft.py`（可重跑）

| 项 | 值 |
| --- | --- |
| 来源 | `runs/*/windows.jsonl` + `decisions.jsonl`（57 次真实闭环运行） |
| 原始窗口 | 140 条，去重后 **87 条** |
| 文件 | `data/sft/policy_sft.jsonl`（154 KB，已入库） |
| 输入 | 窗口的监测/归因结果：任务、主指标与参考指标、漂移统计（KS 中位/最大、\|z\| 中位/最大）、漂移最大的 3 个特征、逐组指标、已用更新次数/epoch |
| 输出 | 当时**真实执行**的动作 + `decisions.jsonl` 里记的真实理由 |
| 动作分布 | hold 66 / finetune_1ep 11 / rollback 7 / finetune_3ep 3 |

> **没有虚构样本**：每条都能回溯到一次运行目录（含 manifest、SwanLab run）。
> 这与前文的"不宣称 A 包提升"是同一条原则——记录里不掺假。

## 3. 训练过程（`harness/trio_sft_policy.py`）

| 项 | 值 |
| --- | --- |
| 接口 | PyTRiO `create_lora_training_client` → `forward_backward` → `optim_step` → `save_weights_for_sampler` |
| 损失 | `cross_entropy` + `weights` 掩码（**只对回答部分计损失**，读题部分权重 0） |
| 批大小 / 轮数 | 8 / 4 epochs → **44 步** |
| 学习率 | Adam 1e-4 |
| 单步耗时 | 约 2.2 s（合计 122.6 s） |

**损失曲线**（真实字段 `metrics.loss_mean`）：

| epoch | 平均 loss | 段内最低 |
| --- | --- | --- |
| 1 | 1.2479 | 0.1778 |
| 2 | 0.2114 | 0.0269 |
| 3 | 0.0635 | 0.0079 |
| 4 | 0.0295 | 0.0005 |

首步 3.1114 → 末步 0.0871，全程下降无发散。完整 44 点曲线见 `runs/trio_sft_report.json`。

## 4. 验证：不是"存下来就算完"

用保存的 Sampler 权重做推理（`harness/trio_verify_sampler.py`），对留出样本比对动作：

| # | 期望 | 得到 | 结果 |
| --- | --- | --- | --- |
| 1 | hold | hold | ✓ |
| 2 | finetune_1ep | finetune_1ep | ✓ |
| 3 | hold | finetune_1ep | ✗ |
| 4 | hold | hold | ✓ |
| 5 | hold | hold | ✓ |

**4/5 命中**，输出是严格可解析的一行 JSON（并以 `<|im_end|>` 收尾），例如：

```
{"action": "hold", "reason": "漂移与指标都在阈值内"}
{"action": "finetune_1ep", "reason": "首次报警：先小步续训试一次"}
```

诚实说明：87 条语料、44 步 → 这是**行为克隆**，训练集上拟合得很好（loss 0.03），
但泛化只能算"可用"。第 3 条说明它在"该 hold"的边界情形上会偏向激进。
所以在正式闭环里它的角色是**建议 + 规则兜底**，不是唯一决策源。

## 5. 复现（两条命令）

```powershell
# 1) 隔离环境装 SDK（依赖里没有 torch，约 1–2 分钟）
python -m venv .trio-env\venv
.\.trio-env\venv\Scripts\python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pytrio transformers modelscope

# 2) 生成语料 → 训练 → 保存 Sampler 权重
python harness\build_policy_sft.py
.\.trio-env\venv\Scripts\python harness\trio_sft_policy.py --epochs 4 --rank 32 --name zhixun-policy-v3
.\.trio-env\venv\Scripts\python harness\trio_verify_sampler.py
```

产物：新 `trio://run_*/sampler_weights/<name>`（脚本会打印）+ `runs/trio_sft_report.json`。

## 6. 订正：之前"只能选基础模型"的结论是错的

上一版清单里写"PyTRiO 仅支持推理、`save_weights_for_sampler` 物理上走不通"。
本轮实测发现：**推理 REST 接口只暴露 2 个 chat 模型，但训练接口（SDK 侧）是可用的**，
`create_lora_training_client` 正常返回，`save_weights_for_sampler` 正常产出
`trio://...` 路径，控制台里能看到类型为 `sampler` 的权重。

当时的误判原因：只探了 OpenAI 兼容的 `/api/openai/v1/models`，看到只有两个模型就下了结论，
没有去试 `TrainingClient`。**记录在这里，避免下次又按"接口不支持"放弃一条可行路径。**

## 7. 与镜像的关系（别被误解）

镜像里的闭环跑在**封闭评测环境**（无外网），所以 `--policy rule` 是默认；
LLM 决策在本地/有端点的环境下用 `--policy auto` 或 `--policy llm` 启用，
走 PyTRiO 的 OpenAI 兼容接口。

这个微调权重的用途是：
1. 表单要求的 **Sampler 持久化权重**（证明真的用了 Sampler 做微调）；
2. 评审可加载它复现"决策模型"的行为（用 §5 第 3 条命令）；
3. 后续把它接进策略层只需要把 `harness/policy.py` 的调用端从 chat 接口换成
   `create_sampling_client(model_path=...)`——**这一步本轮没做**，因为封闭评测无网络，
   做了也无法在评测里生效。
