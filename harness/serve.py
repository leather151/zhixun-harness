"""按官方 `baselines/predict.py` 协议提供推理：stdin 读 JSONL 请求，stdout 写 JSONL 结果。

为什么需要它：平台的提交页只说明"上传 docker save 的镜像"，没有说明隐藏测试数据怎么喂进容器。
官方给的唯一预测协议就是 `predict.py` 这种 **一行请求 → 一行结果** 的管道形式，所以镜像必须支持它。

请求（与官方一致，`task` 可省）：
    {"task": "regression", "inputs": {...}}
响应：
    regression     -> {"outputs": {"value": 12.3}}
    classification -> {"outputs": [{"label": "Ethanol", "score": 0.9}, ...]}

实现说明：真正的加载/预处理/预测逻辑在 `baselines/predict.py`（与 /app/baselines 同一份契约），
这里只做"管道 + 错误兜底"，避免两处实现漂移：
- 自动识别任务（A=128 维 features；B=34 维 horizontal/vertical；C=24 维 settings/sensors）；
- 自动施加检查点自带的预处理（B 包的秩变换分位表）；
- 三条任务流共用一个进程也不会互相污染（vendor 模块按唯一名加载）。
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
for entry in (str(HERE), str(PROJECT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from baselines import predict as BP            # noqa: E402  （BP 是模块，不是函数）


class Server:
    """兼容旧的调用方式：`Server().handle({"inputs": {...}})`。"""

    def __init__(self, model_root=None):
        self.model_root = Path(model_root) if model_root else BP.MODEL_ROOT
        if model_root:
            BP.MODEL_ROOT = self.model_root

    def handle(self, request):
        inputs = request.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("请求必须包含 inputs 对象")
        task = request.get("task")
        if task in ("A", "B", "C"):
            return BP.predict(BP.load(task), inputs)
        return BP.predict_auto(inputs)


def main(argv=None):
    ap = argparse.ArgumentParser(description="官方 predict.py 协议的常驻推理端")
    ap.add_argument("--model-root", default=None, help="默认 models/<A|B|C>/checkpoint*.npz")
    ap.add_argument("--checkpoint", default=None, help="指定单一检查点（仍自动识别任务）")
    ap.add_argument("--task", default=None, choices=["A", "B", "C"], help="固定任务（跳过识别）")
    ap.add_argument("--echo-task", action="store_true", help="在响应里附带识别出的任务名（调试用）")
    args = ap.parse_args(argv)

    # 统一 UTF-8：避免 Windows 上中文类别名/中文路径在管道里乱码，也容忍首行 BOM。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                        # noqa: BLE001
        pass
    stream = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8-sig", errors="replace")

    server = Server(args.model_root)
    fixed = None
    if args.task or args.checkpoint:
        fixed = BP.load(task=args.task, checkpoint=args.checkpoint)

    for line in stream:
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        try:
            request = json.loads(line)
            inputs = request.get("inputs") if isinstance(request, dict) else None
            if not isinstance(inputs, dict):
                raise ValueError("请求必须包含 inputs 对象")
            result = BP.predict(fixed, inputs)
            if args.echo_task:
                result["_task"] = BP.detect_task(BP.flatten_inputs(inputs)[0])
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        except Exception as exc:                              # noqa: BLE001
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False),
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
