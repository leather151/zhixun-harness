# 在我们自己的镜像上重建「SDK 版」交付层
#
# 前置：
#   1) 组委会 base 镜像（image6，含 /app/{engine.py,main.py,pytrio_tools.py,bearing_gate.py,
#      baselines/,configs/} 与 ENTRYPOINT ["python","main.py"]）
#   2) 官方 SDK 包 contest-harness-sdk-1.0.2.tar.gz（放在构建上下文里，命名 sdk.tar.gz）
#   3) 本目录的 contest_submission.py（我们写的 SDK 入口模块）
#
# 用法：
#   docker build -f submission/Dockerfile -t zhixun-harness:sdk --build-arg BASE=<base镜像> .

ARG BASE=zhixun-harness:image6
FROM ${BASE}

# 1) 官方 SDK：解到 /opt 并装进 site-packages（纯标准库，无网络依赖）
ADD sdk.tar.gz /opt/
RUN python3 -c "import sysconfig, shutil; shutil.copytree('/opt/contest-harness-sdk-1.0.2/contest_harness', sysconfig.get_paths()['purelib'] + '/contest_harness')"

# 2) SDK 固定入口：template/harness.py（9 行，调用 contest_harness.cli.main）
COPY harness.py /app/harness.py

# 3) 我们自己的选手模块
COPY contest_submission.py /app/contest_submission.py

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CONTEST_SUBMISSION=/app/contest_submission.py

# 4) 切成 SDK 固定入口：python3 /app/harness.py describe|train|predict
ENTRYPOINT ["python3", "/app/harness.py"]
