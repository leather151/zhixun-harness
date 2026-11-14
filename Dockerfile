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
    HARNESS_DATA=/data \
    HARNESS_VENDOR=/app/vendor \
    HARNESS_MODELS=/app/models \
    PYTHONPATH=/app/harness \
    SWANLAB_MODE=disabled \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

# 官方 CPU 基线代码（可由参赛者修改；保持 ASCII 目录名 A/B/C）
COPY vendor /app/vendor
# 官方起点检查点（父检查点指纹要能被 training.json 对上）
COPY models /app/models
# 闭环系统本体（监测/归因/策略/执行/守门/回滚/日志）
COPY harness /app/harness
# 公开数据（只读副本，仅用于离线自检；正式评测由 /data 挂载覆盖）
COPY data/public /app/data/public

RUN pip install --no-cache-dir -i "$PIP_INDEX" numpy==1.26.4 \
 && python -c "import numpy; print('numpy', numpy.__version__)" \
 && mkdir -p /output /work \
 && useradd -m -u 10001 runner && chown -R runner:runner /output /work

USER runner

# 平台可挂载：/data（隐藏测试数据流）、/output（结果与最终模型落盘）
VOLUME ["/data", "/output"]

# 秒级自检：检查 vendor/models/data 可达，并对每包跑 1 条真实样本推理
HEALTHCHECK --interval=60s --timeout=30s --start-period=10s --retries=3 \
  CMD HARNESS_MODE=selftest python /app/harness/entry.py

# 统一入口：HARNESS_MODE=loop（默认，自进化闭环）| serve（predict 管道）| selftest
ENTRYPOINT ["python", "-u", "/app/harness/entry.py"]
CMD ["--task", "all", "--stream", "auto", "--windows", "4", "--policy", "rule", "--swan", "off", "--out", "/output"]
