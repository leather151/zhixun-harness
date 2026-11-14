"""任务适配层：数据读取、数据流切分、指标分组、模型/训练代码的定位。

设计约束（合规）：
1. **官方 validation 只用于测量**。`--stream official` 模式下窗口来自 validation.jsonl，
   该模式**自动关闭所有更新动作**（updates_allowed=False），只用它演示"漂移监测 + 归因 + 决策"。
2. `--stream train` 模式在官方 train.jsonl 内部按组/时间切出"历史 + 数据流窗口"，
   更新只使用这些（属于训练集内部的切分）。**validation.jsonl 永远不进梯度。**
3. 训练阶段复用的是官方 `baselines/train.py`（`train_run`），保证"续训语义/父检查点/步数"可审计。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

import metrics as M

# ------------------------------------------------------------------ 路径

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent                      # D:\pycharm\competition 或 /app

DEPLOY = os.environ.get("HARNESS_DEPLOY") == "1"   # 容器内


def _first(*cands):
    for c in cands:
        if c and Path(c).exists():
            return Path(c)
    return None


VENDOR_ROOT = _first(os.environ.get("HARNESS_VENDOR"), PROJECT / "vendor")
MODEL_ROOT = _first(os.environ.get("HARNESS_MODELS"), PROJECT / "models")

# 数据根目录候选（按优先级）。注意：**不能只看"目录存在"**——
# 容器里 /data 是声明的 VOLUME，永远存在但是空的，必须检查里面是否真有 train.jsonl。
DATA_CANDIDATES = [os.environ.get("HARNESS_DATA"), PROJECT / "data" / "official",
                   PROJECT / "data" / "public", Path("/app/data/public"), Path("/data"),
                   Path(r"D:\人工智能大赛\02_数据")]
DATA_ROOT = _first(*(c for c in DATA_CANDIDATES if c))


def task_base(task, root=None):
    """定位某任务的数据目录，返回第一个真正含 train.jsonl 的目录。"""
    roots = [root] if root else [c for c in DATA_CANDIDATES if c]
    tried = []
    for r in roots:
        r = Path(r)
        tried.append(str(r))
        for name in (TASKS[task]["dir"], task):        # 中文目录名 / 镜像里的 ASCII 名
            if (r / name / "train.jsonl").exists():
                return r / name
    raise FileNotFoundError(f"找不到 {task} 包数据（试过：{tried}）。"
                            f"可用 HARNESS_DATA 指定数据根目录。")

TASKS = {
    "A": {
        "dir": "A_气体传感器",
        "kind": "classification",
        "classes": ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"],
        "cn": ["乙醇", "乙烯", "氨", "乙醛", "丙酮", "甲苯"],
        "group_of": lambda meta: "class",          # 分组诊断：按真实类别
        "holdout": {"mode": "time_tail_stratified", "share": 0.30},
        "primary": "macro_f1", "higher_is_better": True,
        "unit": "",
    },
    "B": {
        "dir": "B_滚动轴承RUL",
        "kind": "regression",
        "classes": [],
        "cn": [],
        "group_of": lambda meta: "bearing",
        "holdout": {"mode": "group", "groups": ["Bearing2_3", "Bearing3_3"]},
        "primary": "rmse", "higher_is_better": False,
        "unit": "分钟",
    },
    "C": {
        "dir": "C_涡扇发动机RUL",
        "kind": "regression",
        "classes": [],
        "cn": [],
        "group_of": lambda meta: "source_subset",
        "holdout": {"mode": "last_unit_each_subset"},
        "primary": "rmse", "higher_is_better": False,
        "unit": "循环",
    },
}


# ------------------------------------------------------------------ 读取

def flatten_inputs(inputs):
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


def feature_names(paths):
    names = []
    for path in paths:
        parts = json.loads(path)
        if len(parts) >= 3:
            names.append(f"{parts[1][1]}.{parts[2][1]}")
        else:
            group, field = parts[0][1], parts[1][1]
            names.append(f"{group}[{field + 1}]" if isinstance(field, int) else f"{group}.{field}")
    return names


def load_jsonl(path, classes=None):
    ids, x, y, labels, metas, raw = [], [], [], [], [], []
    paths = None
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cur, values = flatten_inputs(row["inputs"])
            if paths is None:
                paths = cur
            ids.append(row["id"])
            metas.append(row.get("meta") or {})
            x.append(values)
            raw.append(line)
            answer = row["answer"]
            if isinstance(answer, str):
                labels.append(answer)
                y.append(classes.index(answer) if classes else -1)
            else:
                labels.append(None)
                y.append(float(answer))
    return {
        "ids": ids, "x": np.asarray(x, dtype=np.float64),
        "y": np.asarray(y, dtype=np.float64), "labels": labels,
        "meta": metas, "paths": paths or [], "raw": raw,
    }


def write_buffer(path, *parts, table=None):
    """把若干行片段写成 train_run 需要的 jsonl。

    `table` 非空时，把每行的输入就地换成秩特征——因为模型的标准化参数是在秩特征上拟合的，
    续训必须喂同样语义的输入，否则等于用错尺子。
    """
    import preprocess
    seen, lines = set(), []
    for part in parts:
        for line, ident in zip(part["raw"], part["ids"]):
            if ident in seen:
                continue
            seen.add(ident)
            lines.append(line if table is None else
                         preprocess.transform_jsonl_line(line, table))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def take(rows, index):
    sub = {k: [v[i] for i in index] for k, v in rows.items()
           if isinstance(v, list) and k != "paths"}
    sub["x"] = rows["x"][index]
    sub["y"] = rows["y"][index]
    sub["paths"] = rows["paths"]
    return sub


def group_values(rows, key):
    if key == "class":
        return np.array(rows["labels"], dtype=object)
    return np.array([str(m.get(key, "?")) for m in rows["meta"]], dtype=object)


# ------------------------------------------------------------------ 数据流

def _chunk(total, windows):
    windows = max(1, min(int(windows), total))
    edges = np.linspace(0, total, windows + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(windows) if edges[i + 1] > edges[i]]


def build_stream(task, mode, windows, data_root=None):
    """返回 {history, windows:[{name,x,y,meta,labels}], updates_allowed, reference}。"""
    spec = TASKS[task]
    base = task_base(task, data_root)
    train = load_jsonl(base / "train.jsonl", spec["classes"] or None)
    valid = load_jsonl(base / "validation.jsonl", spec["classes"] or None)

    if mode == "official":
        chunks = _chunk(len(valid["y"]), windows)
        out = [{"name": f"val-w{i + 1}", **take(valid, list(range(a, b)))}
               for i, (a, b) in enumerate(chunks)]
        return {"history": train, "windows": out, "updates_allowed": False,
                "reference": train, "base": base,
                "note": "official validation：只测不改"}

    # ---- train 内部切分 ----
    hold = spec["holdout"]
    n = len(train["y"])
    if hold["mode"] == "time_tail_stratified":
        # A 包：UCI 是按气体一种一种测的，整段切会让每个窗口只剩 1 个类。
        # 这里改成"每类各取时间尾部 30%，再按时间切成 K 段" → 窗口内类别混合、且仍是时间顺序。
        idx = np.asarray([int(m.get("source_index", i)) for i, m in enumerate(train["meta"])])
        labels = np.asarray(train["labels"], dtype=object)
        hist_idx, hold_parts = [], {}
        for cls in sorted(set(labels.tolist())):
            rows_cls = np.where(labels == cls)[0]
            rows_cls = rows_cls[np.argsort(idx[rows_cls], kind="mergesort")]
            cut = int(round(len(rows_cls) * (1 - hold["share"])))
            hist_idx.extend(rows_cls[:cut].tolist())
            hold_parts[cls] = rows_cls[cut:].tolist()
        hist_idx = np.asarray(sorted(hist_idx))
        per_seg = {cls: _chunk(len(v), windows) for cls, v in hold_parts.items()}
        out = []
        for i in range(windows):
            picks = []
            for cls, parts in per_seg.items():
                if i < len(parts):
                    a, b = parts[i]
                    picks.extend(hold_parts[cls][a:b])
            if picks:
                picks = sorted(picks, key=lambda j: int(train["meta"][j].get("source_index", j)))
                out.append({"name": f"stream-w{i + 1}", **take(train, picks)})
        return {"history": take(train, list(hist_idx)), "windows": out, "updates_allowed": True,
                "reference": take(train, list(hist_idx)), "base": base,
                "note": "train 内部按类别分层的时间切分：可以更新"}
    if hold["mode"] == "time_tail":
        idx = np.asarray([int(m.get("source_index", i)) for i, m in enumerate(train["meta"])])
        order = np.argsort(idx, kind="mergesort")
        cut = int(round(n * (1 - hold["share"])))
        hist_idx, stream_idx = order[:cut], order[cut:]
        stream_rows = take(train, list(stream_idx))
    elif hold["mode"] == "group":
        key = spec["group_of"]({})
        flags = np.array([str(m.get(key, "")) in hold["groups"] for m in train["meta"]])
        hist_idx = np.where(~flags)[0]
        stream_idx = np.where(flags)[0]
        stream_rows = take(train, list(stream_idx))
    elif hold["mode"] == "last_unit_each_subset":
        units = [(m.get("source_subset"), m.get("unit")) for m in train["meta"]]
        last = {}
        for s, u in set(units):
            last[s] = max(last.get(s, -1), u)
        flags = np.array([last.get(s) == u for s, u in units])
        hist_idx = np.where(~flags)[0]
        stream_idx = np.where(flags)[0]
        stream_rows = take(train, list(stream_idx))
    else:
        raise ValueError("unknown holdout mode")

    history = take(train, list(hist_idx))
    chunks = _chunk(len(stream_rows["y"]), windows)
    out = []
    for i, (a, b) in enumerate(chunks):
        out.append({"name": f"stream-w{i + 1}", **take(stream_rows, list(range(a, b)))})
    return {"history": history, "windows": out, "updates_allowed": True,
            "reference": history, "base": base,
            "note": f"train 内部切分（{hold['mode']}）：可以更新"}


# ------------------------------------------------------------------ 指标

def evaluate(spec, model, rows, paths_names=None):
    pred_raw = model.predict(model_inputs(model, rows["x"]))
    if spec["kind"] == "classification":
        pred = np.asarray(pred_raw).argmax(1)
        metrics_ = {
            "macro_f1": M.macro_f1(rows["y"].astype(int), pred, len(spec["classes"])),
            "accuracy": float(np.mean(pred == rows["y"])),
        }
        conf = M.confusion(rows["y"].astype(int), pred, len(spec["classes"]))
        groups = group_values(rows, "class")
        by_group = {}
        for c, name in enumerate(spec["cn"]):
            sel = rows["y"].astype(int) == c
            if sel.sum():
                by_group[name] = {"n": int(sel.sum()),
                                  "recall": float(np.mean(pred[sel] == c))}
        extra = {"confusion": conf.tolist(),
                 "confidence_mean": float(np.asarray(pred_raw).max(1).mean())}
    else:
        pred = np.asarray(pred_raw, dtype=np.float64)
        metrics_ = {"rmse": M.rmse(pred, rows["y"]), "mae": M.mae(pred, rows["y"]),
                    "bias": M.bias(pred, rows["y"])}
        if spec is TASKS["C"]:
            metrics_["nasa"] = M.nasa_score(pred, rows["y"])
        groups = group_values(rows, spec["group_of"]({}))
        by_group = {}
        for g in sorted(set(groups.tolist())):
            sel = groups == g
            by_group[str(g)] = {"n": int(sel.sum()), "rmse": M.rmse(pred[sel], rows["y"][sel]),
                                "mae": M.mae(pred[sel], rows["y"][sel]),
                                "bias": M.bias(pred[sel], rows["y"][sel])}
        extra = {}
    return {"metrics": metrics_, "by_group": by_group, "pred": pred, "raw": pred_raw, **extra}


# ------------------------------------------------------------------ 官方代码

def load_vendor(task, need_train=False):
    """把 vendor/<task> 加入 sys.path，导入官方的 model / train 模块。"""
    vdir = VENDOR_ROOT / task
    if not (vdir / "model.py").exists():
        raise FileNotFoundError(f"缺少官方基线代码：{vdir}")
    tag = getattr(load_vendor, "_tag", None)
    if tag and tag != task:
        raise RuntimeError(f"同一进程不能混用两个任务的 vendor 代码（已有 {tag}，又请求 {task}）")
    load_vendor._tag = task
    if str(vdir) not in sys.path:
        sys.path.insert(0, str(vdir))
    import importlib
    model_mod = importlib.import_module("model")
    out = {"model": model_mod, "dir": vdir}
    if need_train:
        out["train"] = importlib.import_module("train")
        out["data"] = importlib.import_module("data")
    return out


def checkpoint_path(task, name=None):
    """默认起点/交付模型：优先用优化后的检查点，其次是官方起点。

    优先级：checkpoint_best.npz → checkpoint_rank.npz（含秩分位表的改进版）→ checkpoint.npz（官方起点）
    """
    root = MODEL_ROOT / task
    if name:
        return root / name
    for candidate in ("checkpoint_best.npz", "checkpoint_rank.npz", "checkpoint.npz"):
        if (root / candidate).exists():
            return root / candidate
    return root / "checkpoint.npz"


def model_table(model):
    """取模型自带的预处理分位表（没有则 None）。"""
    import preprocess
    return preprocess.table_from_meta(model.meta)


def model_inputs(model, x):
    """按模型自带预处理变换输入（秩变换模型必须先变换，否则等于喂错数据）。"""
    import preprocess
    return preprocess.transform_inputs(model.meta, x)


def load_model(task, path):
    mods = load_vendor(task)
    return mods["model"].Model.load(Path(path))


def fine_tune(task, parent, train_path, validation_path, out_dir, epochs, learning_rate,
              profile=None, stage_id="harness"):
    """调用官方 train_run 续训一轮；返回 (report_dict, out_dir)。"""
    mods = load_vendor(task, need_train=True)
    report = mods["train"].train_run(
        profile or task, str(train_path), str(validation_path), str(out_dir),
        resume=str(parent), epochs=int(epochs), learning_rate=learning_rate, stage_id=stage_id)
    return report, Path(out_dir)
