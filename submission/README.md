# submission/ —— SDK 版镜像里**我们自己的那一层**

## 1. 这份目录是什么

`image6-sdk-linux-amd64.tar` 是在**组委会 image6 底座 + 官方 contest-harness SDK** 之上，
由我们重新构建的一层镜像。本目录只收录**我们自己写的部分**，不随仓库分发组委会的实现。

| 部分 | 归属 | 位置 |
| --- | --- | --- |
| `contest_submission.py` | **我们（SDK 入口模块）** | `app/contest_submission.py` |
| 构建步骤（ADD SDK + COPY 我们的模块 + 改 ENTRYPOINT） | **我们** | 见 §2 |
| `engine.py` / `main.py` / `pytrio_tools.py` / `bearing_gate.py` / `baselines/` / `configs/` | 组委会 image6 底座 | 不收录 |
| `contest_harness` SDK | 组委会（`contest-harness-sdk-1.0.2`） | 不收录 |

## 2. 我们的构建层（镜像 history 的最后 5 步，`created: 2026-11-17T00:20:52Z`）

```
ADD sdk.tar.gz /opt/                                                    # 官方 SDK 包
RUN python3 -c "import sysconfig,shutil; shutil.copytree(
        '/opt/contest-harness-sdk-1.0.2/contest_harness',
        sysconfig.get_paths()['purelib'] + '/contest_harness')"          # 安装进 site-packages
COPY contest_submission.py /app/contest_submission.py                    # 我们的桥接模块
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["python3", "/app/harness.py"]                                # 切成 SDK 固定入口
```

重建方式（`image6` 指组委会给的底座镜像）：

```dockerfile
FROM <官方 image6 镜像>
ADD sdk.tar.gz /opt/
RUN python3 -c "import sysconfig,shutil; shutil.copytree('/opt/contest-harness-sdk-1.0.2/contest_harness', sysconfig.get_paths()['purelib'] + '/contest_harness')"
COPY contest_submission.py /app/contest_submission.py
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["python3", "/app/harness.py"]
```

## 3. `contest_submission.py` 做了什么

严格按官方 SDK 1.0.2 / 协议 `contest.harness.workspace.v2` 实现两个入口：

### `train(rnd)` —— 一轮训练

1. 读 `rnd.dataset_key`（A/B/C）与 `configs/<key>.json`，核对 `rnd.task["task"]` 与配置一致；
2. **父模型**取 `rnd.parent.path("checkpoint.npz")` 并算摘要（SDK 自带摘要校验）；
3. **真实模型调用**（SDK 强制要求，且 `finish` 的 `model_call_ids` 不能为空）：
   把「指标、最近 3 轮 reason、两组固定试验、两个候选第三组」发给大模型，
   要求它只回 `conservative` 或 `standard`；
4. 用官方 `baselines.train.train_run` **从同一个父检查点跑 3 组试验**：
   - `short` = (lr 2.5e-4, epochs 1, batch 256)
   - `regular` = (lr 1e-3, epochs 3, batch 128)
   - 第三组 = 模型选中的 `conservative` (5e-4, 2, 256) 或 `standard` (1.5e-3, 3, 128)
5. 每组校验「确实是父检查点的子代」：
   `parent_checkpoint_sha256` 匹配、`final_steps > initial_steps`、`checkpoint_sha256 != 父摘要`，
   否则抛错（防止静默沿用父模型冒充训练结果）；
6. 按验证集指标选最优一组 → `rnd.submit_candidate(...)` 提交（`predict_argv=["python3","/app/harness.py","predict"]`）；
7. `rnd.finish(selected_artifact_id = 子代 if 优于父 否则 父)`，并在 `reason` 里写清
   `adopt_child` / `keep_existing_parent` 及父/子两侧指标。

### `predict(sample, model_dir)`

1. `model_dir` 是 `/model` 根目录，检查点是 `model_dir/checkpoint.npz`（`load_model` 带 LRU 缓存）；
2. 校验 `sample.task` 与检查点 `task` 一致、`sample.dataset_key` 与 `profile` 一致，不一致直接抛错；
3. 复用官方 `baselines.predict.predict(model, sample.inputs)["outputs"]` 输出
   `{"value": …}` 或 `[{"label":…, "score":…}]`。

## 4. 为什么这样分层

- **SDK 负责协议**（describe/train/predict、候选提交、状态、回放、模型调用的密钥与计费），
  我们不重复实现，也不把 PyTRiO 密钥写进镜像（SDK 文档明确要求）。
- **我们负责算法**：三组试验的配方、父代校验、按验证集选优、采纳/保留的判定与理由。
- 采纳与否最终由 `rnd.finish(selected_artifact_id=...)` 声明，**平台测量后返回真实摘要**，
  我们不在代码里自报分数。

## 5. 本地验证方式

```bash
# 用 SDK 自带的离线单测（JSONL 宿主替身，不调用真实模型）
cd <sdk>/contest-harness-sdk-1.0.2
python -m unittest tests.test_sdk tests.test_contract_regressions tests.test_transport_regressions -v
# 实测：15 个测试全部通过
```
