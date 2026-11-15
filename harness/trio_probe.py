"""探测 PyTRiO 账号能力：能否登录、有没有已有权重、能否创建 LoRA 训练客户端。

只做只读探测（列权重 / 列训练运行）+ 创建训练客户端（不提交训练步），不消耗训练额度。
"""
import os
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent


def load_key():
    if os.environ.get("PYTRIO_API_KEY"):
        return os.environ["PYTRIO_API_KEY"]
    env = APP / ".env.local"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("PYTRIO_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("找不到 PYTRIO_API_KEY（环境变量或 .env.local）")


os.environ["PYTRIO_API_KEY"] = load_key()

import pytrio as trio  # noqa: E402

print("pytrio", getattr(trio, "__version__", "?"))

client = trio.ServiceClient()
print("[1] ServiceClient OK")

models = client.get_supported_models()
print("[2] 支持模型:", models)

rest = client.create_rest_client()
try:
    ck = rest.list_user_checkpoints(limit=50, offset=0).result()
    print("[3] 已有权重:", ck)
except Exception as exc:                                     # noqa: BLE001
    print("[3] 列权重失败:", type(exc).__name__, exc)

try:
    runs = rest.list_training_runs(limit=10, offset=0).result()
    print("[4] 已有训练运行:", runs)
except Exception as exc:                                     # noqa: BLE001
    print("[4] 列训练运行失败:", type(exc).__name__, exc)
