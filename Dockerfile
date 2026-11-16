# 智训引擎 · 工业模型自进化系统镜像
#
# 构建（Windows / PowerShell）：  .\build.ps1
# 构建（任意平台）：              docker build -t zhixun-harness:dev .
#
# 设计原则：
# 1) **纯 CPU、无网络、无外部依赖**：只装 numpy，SwanLab 只在本地开发用，镜像里不装；
# 2) **受限机制**：窗口数 / 更新次数 / epoch / 秒数都有硬上限（harness/config.json 的 caps）；
# 3) **数据与代码分离**：/data 由平台挂载，缺省回落到镜像内的公开数据（/app/data/public），
#    这样镜像自身可离线自检。
FROM python:3.12-slim

# PyPI 源可替换（国内构建时用清华源）：docker build --build-arg PIP_INDEX=...
ARG PIP_INDEX=https://pypi.org/simple

LABEL org.opencontainers.image.title="zhixun-engine-self-evolving-harness" \
      org.opencontainers.image.description="漂移监测-归因-实验优化-评估上线-回滚 闭环（CPU/离线）" \
      org.opencontainers.image.version="0.1.0"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    HARNESS_DEPLOY=1 \
    HARNESS_START_POLICY=previous \
    HARNESS_DATA=/data \
    HARNESS_VENDOR=/app/vendor \
    HARNESS_MODELS=/app/models \
    PYTHONPATH=/app:/app/harness \
    SWANLAB_MODE=disabled \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# 官方 CPU 基线代码（可由参赛者修改；保持 ASCII 目录名 A/B/C）
COPY vendor /app/vendor
# 官方起点检查点 + 每个任务的纯 JSON 摘要（models/<T>/meta.json，预检直接读）
COPY models /app/models
# 官方风格预测桥：baselines.predict.predict(model, inputs)（含 B 包的秩变换链路）
COPY baselines /app/baselines
# 闭环系统本体（监测/归因/策略/执行/守门/回滚/日志）
COPY harness /app/harness
# 适配契约：ADAPTER.md 给人看，CONTRACT.json 给机器读（预检只需这两个 + models/<T>/meta.json）
COPY ADAPTER.md CONTRACT.json /app/
# 公开数据（只读副本，仅用于离线自检；正式评测由 /data 挂载覆盖）
COPY data/public /app/data/public
# PyTRiO SDK 的离线 wheelhouse（仅构建期使用，装完即删，不留在运行镜像里）
COPY wheelhouse /app/wheelhouse

RUN pip install --no-cache-dir -i "$PIP_INDEX" numpy==1.26.4 \
 && python -c "import numpy; print('numpy', numpy.__version__)" \
 && PYTHONPATH=/app:/app/harness python -c "from baselines.predict import predict; print('baselines.predict OK')" \
 && mkdir -p /output /work /model \
 && useradd -m -u 10001 runner && chown -R runner:runner /output /work /model

# ---- 路线二（可选）：PyTRiO SDK 放进独立 venv，**离线安装**，绝不污染主运行时 ----
# 为什么隔离：transformers/modelscope 会要求 numpy 2.x，而主运行时锁在 1.26.4（那套已验证）。
# 为什么在镜像里：评测环境断网，只有装进来的东西才用得上；用自己微调的 sampler 权重
# （trio://run_.../sampler_weights/...）必须靠 SDK 的 create_sampling_client(model_path=...)。
ARG INSTALL_TRIO_SDK=1
ARG TOKENIZER_MODEL=
RUN if [ "$INSTALL_TRIO_SDK" = "1" ] && [ -d /app/wheelhouse ]; then \
      python -m venv /app/.trio-venv \
      && /app/.trio-venv/bin/pip install --no-index --no-cache-dir --no-compile \
           --find-links /app/wheelhouse pytrio transformers \
      && /app/.trio-venv/bin/python -c "import pytrio, transformers; print('trio venv OK')" \
      && if [ -n "$TOKENIZER_MODEL" ]; then \
           echo "[info] 不预取 tokenizer（$TOKENIZER_MODEL）；运行时可从平台缓存或 HARNESS_TRIO_CACHE 读取"; \
         fi; \
    else echo '[skip] 不安装 PyTRiO SDK'; fi \
 && rm -rf /app/wheelhouse

ENV HARNESS_TRIO_VENV_PYTHON=/app/.trio-venv/bin/python \
    HARNESS_TRIO_CACHE=/opt/trio-cache

USER runner

# 平台可挂载：/data（隐藏测试数据流）、/output（结果落盘）、/previous（父检查点）、/model（交付模型）
VOLUME ["/data", "/output", "/previous", "/model"]

# 秒级自检：检查 vendor/models/data 可达，并对每包跑 1 条真实样本推理
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
  CMD HARNESS_MODE=selftest python /app/harness/entry.py

# 统一入口：HARNESS_MODE=loop（默认，自进化闭环）| serve（predict 管道）| selftest
ENTRYPOINT ["python", "-u", "/app/harness/entry.py"]
# 默认跑闭环：stream=train 才允许"读历史→做实验→上线/回滚"，官方 validation 始终只测不改；
# policy=auto：评测环境若由平台注入 PyTRiO 接入信息（SAMPLER_* / PYTRIO_*），会自动改用
# Qwen3.8-27B 作为"合规驱动模型"；没有注入则回落确定性规则版，保证"无人干预跑完"这条硬门槛。
CMD ["--task", "all", "--stream", "train", "--windows", "4", "--policy", "auto", "--swan", "off", "--out", "/output"]
