"""官方基线风格的预测入口：`from baselines.predict import predict`。

为什么单独建这个包（而不是直接用 `vendor/<TASK>/predict.py`）：
1. 官方三个包的 `predict.py` 各管一个任务，**同一个进程里混用会互相污染**
   （它们都以顶层模块名 `model` 导入，后导入的会拿到前一个任务的缓存）；
2. 我们 B 包的检查点带**秩变换分位表**（`metadata.preprocess`），官方 `predict.py`
   不做这一步变换，直接喂原始特征等于用错尺子（RMSE 会从 369 掉到 ~1400）。
   这里统一走 `harness/preprocess.py` 的同一份实现，保证离线评估与线上推理一致。

对外的契约（与官方 `predict(model, inputs)` 同形）：

    predict(model, inputs) -> dict
        model : 已加载的 Model 对象 / 检查点路径 / None（None 表示按输入自动识别任务）
        inputs: 一行输入（嵌套 dict，例如 {"features": [...]}）
        返回   : 回归  {"outputs": {"value": 12.3}}
                 分类  {"outputs": [{"label": "...", "score": 0.9}, ...]}

辅助入口：
    load(task=None, checkpoint=None)   取（并缓存）某任务的模型
    predict_auto(inputs)               按输入 schema 自动识别任务后预测
    main()                             JSONL 管道：stdin 读 {"inputs": ...} → stdout 写结果
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
APP = HERE.parent
HARNESS = APP / "harness"
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

import preprocess as PRE  # noqa: E402

VENDOR_ROOT = Path(__file__).resolve().parent.parent / "vendor"
MODEL_ROOT = Path(__file__).resolve().parent.parent / "models"

# 每个任务的检查点优先级（与 harness/tasks.checkpoint_path 保持一致）
# 交付位优先：平台读 checkpoint.npz 且用自己的原始特征推理路径，所以它必须是"开箱即用"的模型
CHECKPOINT_ORDER = ("checkpoint.npz", "checkpoint_best.npz", "checkpoint_rank.npz")

_MODEL_MODULES = {}
_MODELS = {}


# ------------------------------------------------------------------ 任务识别

def detect_task(paths):
    """按扁平化后的字段路径识别 A/B/C；识别不了就用特征维度兜底。"""
    joined = "|".join(str(p) for p in paths)
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


def flatten_inputs(inputs):
    """与官方一致的扁平化：dict 的键按字母序、list 按下标。返回 (paths, values)。"""
    paths, values = [], []

    def visit(value, path):
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], path + [("key", key)])
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [("index", index)])
        else:
            paths.append(json.dumps(path, ensure_ascii=False, separators=(",", ":")))
            values.append(float(value))

    visit(inputs, [])
    return paths, values


# ------------------------------------------------------------------ 模型加载

def _model_module(task):
    """按任务加载 `vendor/<TASK>/model.py`，用**唯一模块名**，避免多任务互相覆盖。"""
    if task in _MODEL_MODULES:
        return _MODEL_MODULES[task]
    path = VENDOR_ROOT / task / "model.py"
    if not path.is_file():
        raise FileNotFoundError(f"缺少基线代码：{path}")
    name = f"_zhixun_vendor_{task}_model"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _MODEL_MODULES[task] = module
    return module


def default_checkpoint(task):
    root = MODEL_ROOT / task
    for name in CHECKPOINT_ORDER:
        cand = root / name
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"缺少模型：{root}/checkpoint*.npz")


# ------------------------------------------------------------------ 按 schema 找模型
# 为什么需要：评测数据包**不一定**就是 gas / 轴承 / C-MAPSS 这三套。
# 一旦输入的特征字段/维度变了，按"任务名"找模型会直接失败。
# 这里改成**按 schema 匹配**：扫一遍所有检查点，谁的 `metadata.paths` 与本次输入
# 逐字段一致就用谁 —— 这样只要该数据上训练过一次模型，预测就能接上。

_SCHEMA_INDEX = {}


def _candidate_dirs():
    """检查点搜索目录：镜像内、平台父模型、交付目录、以及环境变量指定的位置。"""
    dirs = [MODEL_ROOT, Path("/previous"), Path("/model"), Path("/output/model")]
    for extra in (os.environ.get("HARNESS_MODELS"), os.environ.get("HARNESS_PREVIOUS"),
                  os.environ.get("HARNESS_MODEL_OUT")):
        if extra:
            dirs.append(Path(extra))
    return [d for d in dirs if d and Path(d).is_dir()]


def checkpoint_pool():
    """所有可用的检查点（镜像内 + 平台挂载的父模型）。"""
    seen, pool = set(), []
    for root in _candidate_dirs():
        for path in sorted(Path(root).rglob("*.npz")):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            pool.append(path)
    return pool


def schema_index(refresh=False):
    """{paths 元组: 检查点路径}，用于按输入 schema 精确匹配。"""
    global _SCHEMA_INDEX
    if _SCHEMA_INDEX and not refresh:
        return _SCHEMA_INDEX
    index = {}
    for path in checkpoint_pool():
        try:
            model = load(checkpoint=path)
            key = tuple(model.meta.get("paths") or ())
            if key:
                index.setdefault(key, path)
        except Exception:                                     # noqa: BLE001
            continue
    _SCHEMA_INDEX = index
    return index


def find_by_schema(paths):
    """按输入 schema 找匹配的检查点；找不到返回 None。"""
    key = tuple(paths)
    path = schema_index().get(key)
    if path is None:
        path = schema_index(refresh=True).get(key)   # 可能刚训练出新的检查点
    return path


def load_for_schema(paths):
    """给一个输入 schema，返回可直接预测的模型。

    顺序：内置 A/B/C（schema 一致时）→ 任意扫描到的同 schema 检查点 → 报错。
    """
    try:
        task = detect_task(paths)
    except ValueError:
        task = None
    if task is not None:
        try:
            model = load(task)
            if tuple(model.meta.get("paths") or ()) == tuple(paths):
                return model
        except Exception:                                     # noqa: BLE001
            pass
    path = find_by_schema(paths)
    if path is not None:
        return load(checkpoint=path)
    raise ValueError(
        "输入 schema 与镜像内任何检查点都不匹配，且该数据上还没有训练过模型。"
        f"（{len(paths)} 个特征；已扫描 {len(checkpoint_pool())} 个检查点）"
        "请先用 `python -m baselines.train` 在该数据上训练一轮，或提供 HARNESS_MODELS/HARNESS_PREVIOUS。")


def load(task=None, checkpoint=None):
    """加载（并缓存）某个任务的模型。`checkpoint` 给了就以它为准，任务从其 metadata 反推。"""
    if checkpoint is not None:
        path = Path(checkpoint)
        if path in _MODELS:
            return _MODELS[path]
        # 任务未知时依次尝试三个 vendor，第一个加载成功即用
        candidates = [task] if task else ["A", "B", "C"]
        last = None
        for name in candidates:
            try:
                model = _model_module(name).Model.load(path)
            except Exception as exc:                          # noqa: BLE001
                last = exc
                continue
            _MODELS[path] = model
            return model
        raise ValueError(f"无法加载检查点 {path}：{type(last).__name__}: {last}")
    if task not in _MODELS:
        _MODELS[task] = _model_module(task).Model.load(default_checkpoint(task))
    return _MODELS[task]


# ------------------------------------------------------------------ 预测

def predict(model, inputs):
    """官方 `predict(model, inputs)` 同形入口。model 可为 Model / 路径 / None。"""
    if model is None:
        return predict_auto(inputs)
    if isinstance(model, (str, Path)):
        model = load(checkpoint=model)
    if not isinstance(inputs, dict):
        raise ValueError("inputs 必须是对象")

    paths, values = flatten_inputs(inputs)
    meta_paths = list(model.meta.get("paths") or [])
    if len(meta_paths) != len(paths):
        raise ValueError(f"特征维度不一致：输入 {len(paths)} 维，检查点 {len(meta_paths)} 维")
    if paths != meta_paths:
        # 字段顺序不同但维度相同：按检查点自己记录的顺序重排，而不是直接报错
        pos = {p: i for i, p in enumerate(paths)}
        if set(pos) != set(meta_paths):
            raise ValueError("特征 schema 与检查点不一致（字段名不同）")
        values = [values[pos[p]] for p in meta_paths]

    x = np.asarray([values], dtype=np.float64)
    # 关键：按检查点自带的预处理变换（B 包的秩变换在这里生效）
    x = PRE.transform_inputs(model.meta, x)
    out = np.asarray(model.predict(x))[0]

    if model.meta.get("task") == "regression":
        return {"outputs": {"value": float(out)}}
    return {"outputs": [{"label": label, "score": float(score)}
                        for label, score in zip(model.meta.get("classes") or [], out)]}


def predict_auto(inputs):
    """按输入 schema 自动选模型并预测（镜像里 `HARNESS_MODE=serve` 走的是同一条逻辑）。

    先按内置 A/B/C 识别；识别不出来（说明是**新的数据包**）就按 schema
    在镜像与 /previous 里找同字段的检查点 —— 只要该数据上训练过一轮就能接上。
    """
    paths, _ = flatten_inputs(inputs)
    return predict(load_for_schema(paths), inputs)


# ------------------------------------------------------------------ CLI

def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="官方 predict 协议的桥接实现")
    ap.add_argument("--checkpoint", default=None, help="检查点路径；省略则按输入自动选任务")
    ap.add_argument("--task", default=None, choices=["A", "B", "C"])
    args = ap.parse_args(argv)

    model = None
    if args.checkpoint or args.task:
        model = load(task=args.task, checkpoint=args.checkpoint)

    for line in sys.stdin:
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        try:
            request = json.loads(line)
            inputs = request.get("inputs") if isinstance(request, dict) else None
            if not isinstance(inputs, dict):
                raise ValueError("请求必须包含 inputs 对象")
            result = predict(model, inputs)
            print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        except Exception as exc:                              # noqa: BLE001
            print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False),
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
