"""按官方 `baselines/predict.py` 协议提供推理：stdin 读 JSONL 请求，stdout 写 JSONL 结果。

为什么需要它：平台的提交页只说明"上传 docker save 的镜像"，没有说明隐藏测试数据怎么喂进容器。
官方给的唯一预测协议就是 `predict.py` 这种 **一行请求 → 一行结果** 的管道形式，所以镜像必须支持它。

请求（与官方一致，`task` 可省）：
    {"task": "regression", "inputs": {...}}
响应：
    regression     -> {"outputs": {"value": 12.3}}
    classification -> {"outputs": [{"label": "Ethanol", "score": 0.9}, ...]}

自动识别任务（按特征维度/字段名）：A=128 维 features；B=34 维（horizontal/vertical）；C=24 维（settings/sensors）。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import tasks as T   # noqa: E402

PROJECT = HERE.parent


def detect_task(paths):
    """按官方 flatten 出来的字段路径识别任务。"""
    joined = "|".join(paths)
    n = len(paths)
    if "horizontal" in joined and "vertical" in joined:
        return "B"
    if "settings" in joined and "sensors" in joined:
        return "C"
    if "features" in joined and n == 128:
        return "A"
    if n == 128:
        return "A"
    if n == 34:
        return "B"
    if n == 24:
        return "C"
    raise ValueError(f"无法识别的输入结构（{n} 个特征）：{joined[:200]}")


class Server:
    def __init__(self, model_root=None):
        self.model_root = Path(model_root) if model_root else T.MODEL_ROOT
        self._models = {}

    def model(self, task):
        if task not in self._models:
            import importlib
            vdir = T.VENDOR_ROOT / task
            if str(vdir) not in sys.path:
                sys.path.insert(0, str(vdir))
            mod = importlib.import_module("model")
            ckpt = self.model_root / task / "checkpoint.npz"
            if not ckpt.exists():
                raise FileNotFoundError(f"缺少模型：{ckpt}")
            self._models[task] = mod.Model.load(ckpt)
        return self._models[task]

    def handle(self, request):
        inputs = request.get("inputs")
        if not isinstance(inputs, dict):
            raise ValueError("请求必须包含 inputs 对象")
        paths, values = T.flatten_inputs(inputs)
        task = detect_task(paths)
        model = self.model(task)
        if paths != model.meta["paths"]:
            raise ValueError(f"特征 schema 与检查点不一致（识别为 {task} 包）")
        out = model.predict(np.asarray([values], dtype=np.float64))[0]
        if model.meta["task"] == "regression":
            return {"outputs": {"value": float(out)}}
        return {"outputs": [{"label": label, "score": float(score)}
                            for label, score in zip(model.meta["classes"], out)]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="官方 predict.py 协议的常驻推理端")
    ap.add_argument("--model-root", default=None, help="默认 models/<A|B|C>/checkpoint.npz")
    ap.add_argument("--echo-task", action="store_true", help="在响应里附带识别出的任务名（调试用）")
    args = ap.parse_args(argv)

    # 统一 UTF-8：避免 Windows 上中文类别名/中文路径在管道里乱码，也容忍首行 BOM。
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:                                        # noqa: BLE001
        pass
    stream = io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8-sig", errors="replace")

    server = Server(args.model_root)
    for line in stream:
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        try:
            request = json.loads(line)
            result = server.handle(request)
            if args.echo_task:
                result["_task"] = detect_task(T.flatten_inputs(request.get("inputs", {}))[0])
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        except Exception as exc:                              # noqa: BLE001
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False),
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
