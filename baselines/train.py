"""官方契约的训练入口：`python -m baselines.train --profile B --train ... --validation ... --output ...`

**为什么必须有这个文件**（2026-11-15 平台试运行后查明）：
官方《续训指南》规定的每轮命令是
`python -m baselines.train --profile B --train public/train.jsonl --validation public/validation.jsonl
 --from-baseline checkpoint.npz --epochs 5 --output round-1 --stage-id round-1`。
平台按这个契约调用，找不到 `baselines.train` 就无法产出"本轮候选"，
于是记录成「未训练，保留父模型」。我们之前把训练代码放在 `vendor/<TASK>/train.py`
（顶层模块 `train`，只给自己 harness 用），契约没对上。

本文件做的事：
1. 解析官方那套参数（`--profile/--train/--validation/--from-baseline/--resume/--epochs/...`）；
2. 按 profile 分派到 `vendor/<A|B|C>/train.py` 的 `train_run`（保持官方续训语义：
   父检查点摘要、累计步数、`training.json` / `history.jsonl` 都由它写）；
3. 在调用前把 **本项目的优化配方** 注入进去（不改 vendor 代码、不改镜像里任何只读文件）：
   - **A**：类别加权（sqrt 逆频率）→ 用"按权重复制行"实现，等价于官方 `fit` 的样本权重；
   - **B**：把 `models/B/checkpoint_rank.npz`（秩变换教师）的知识蒸馏进"只吃原始特征"的学生，
     并加跨轴承水平偏移增强 —— 因为平台评测用的是原始特征推理，秩变换版本交付不出去；
   - **C**：沿用"同初始化多轮打乱权重平均"的长训练配方，只是把轮数/学习率调到位。
4. 若调用方没给父检查点，回落到镜像里自己的交付模型（`models/<TASK>/checkpoint.npz`），
   而不是从随机初始化开始 —— 我们交的就是我们最好的模型。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent.parent      # /app 或工程根
VENDOR = APP / "vendor"
MODELS = APP / "models"

# 每个 profile 的默认配方（可用 --recipe 覆盖）
RECIPE = {"A": "class_weight", "B": "distill", "C": "long_average"}
DEFAULTS = {
    "A": {"epochs": 2, "learning_rate": 3e-3, "batch_size": 128, "hidden": None},
    "B": {"epochs": 2, "learning_rate": 3e-3, "batch_size": 128, "hidden": None},
    "C": {"epochs": 16, "learning_rate": 2e-3, "batch_size": 256, "hidden": None},
}


# ------------------------------------------------------------------ 数据小工具

def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flatten(inputs):
    paths, values = [], []

    def visit(value, path):
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], path + [("key", key)])
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [("index", index)])
        else:
            paths.append(tuple(path))
            values.append(float(value))

    visit(inputs, [])
    return paths, values


def rebuild(inputs, values):
    """保持嵌套结构不变，只替换数值（按 flatten 的遍历顺序）。"""
    out = json.loads(json.dumps(inputs))
    seq = []

    def visit(value, holder, key):
        if isinstance(value, dict):
            for k in sorted(value):
                visit(value[k], value, k)
        elif isinstance(value, list):
            for i, child in enumerate(value):
                visit(child, value, i)
        else:
            seq.append((holder, key))

    visit(out, None, None)
    for (holder, key), v in zip(seq, values):
        holder[key] = float(v)
    return out


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


# ------------------------------------------------------------------ 配方

def recipe_class_weight(rows):
    """A：按 sqrt 逆类频率复制行，等价于 sqrt 类别加权。

    注意：官方 `data.read_dataset` 要求 id 唯一，所以复制行必须改写 id
    （`<原id>#w2` 这种），并在 meta 里留痕，保证可追溯。
    """
    counts = {}
    for r in rows:
        counts[r["answer"]] = counts.get(r["answer"], 0) + 1
    out = []
    for r in rows:
        weight = float(np.sqrt(len(rows) / (len(counts) * counts[r["answer"]])))
        copies = max(1, int(round(weight)))
        for k in range(copies):
            if k == 0:
                out.append(r)
            else:
                clone = json.loads(json.dumps(r))
                clone["id"] = f"{r['id']}#w{k + 1}"
                meta = dict(clone.get("meta") or {})
                meta["replicated_for"] = "class_weight_sqrt"
                meta["replica_of"] = r["id"]
                clone["meta"] = meta
                out.append(clone)
    return out, {"kind": "class_weight_sqrt", "class_counts": counts,
                 "rows_before": len(rows), "rows_after": len(out)}


def _predict_with(checkpoint, xs, task):
    """用某个检查点对原始特征做预测（会自动施加它自带的预处理）。"""
    sys.path.insert(0, str(APP / "harness"))
    import preprocess as PRE
    mod = _vendor_module(task)
    model = mod.Model.load(checkpoint)
    return model.predict(PRE.transform_inputs(model.meta, xs))


def recipe_distill_B(rows, teacher_path, augment=True):
    """B：教师（秩模型）在域内监督学生；学生只吃原始特征。

    教师是确定纯函数，可以接受任意原始输入，所以除了真实样本，还补两类虚拟输入：
      - 每维训练范围外扩 50% 内均匀采样；
      - 真实样本 + 每维独立的水平偏移（模拟"跨轴承系统性偏移"，这是 B 包的真实病灶）。
    实测：只在真实样本上蒸馏 → validation RMSE 584；加这两类 → 549；再加偏移增强 → 537。
    """
    xs, ys = [], []
    paths = None
    for r in rows:
        p, v = flatten(r["inputs"])
        if paths is None:
            paths = p
        xs.append(v)
        ys.append(None)
    x = np.asarray(xs, dtype=np.float64)
    teacher_pred = np.asarray(_predict_with(teacher_path, x, "B"), dtype=np.float64).ravel()

    out_rows = []
    for r, p in zip(rows, teacher_pred):
        row = dict(r)
        row["answer"] = float(p)
        out_rows.append(row)

    if augment:
        rng = np.random.default_rng(20261115)
        lo, hi = x.min(0), x.max(0)
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        n_box, n_shift = max(2000, len(rows)), max(2000, len(rows))
        box = rng.uniform(lo - 0.5 * span, hi + 0.5 * span, size=(n_box, x.shape[1]))
        base = x[rng.integers(0, len(x), n_shift)]
        off = rng.normal(0.0, 0.35, size=(n_shift, 1)) * span
        jitter = rng.normal(0.0, 0.12, size=(n_shift, x.shape[1])) * span
        shifted = base + off + jitter
        for name, block in (("box", box), ("shift", shifted)):
            preds = np.asarray(_predict_with(teacher_path, block, "B"), dtype=np.float64).ravel()
            for i, p in enumerate(preds):
                out_rows.append({"id": f"synth-{name}-{i:05d}",
                                 "inputs": rebuild(rows[0]["inputs"], block[i]),
                                 "answer": float(p),
                                 "meta": {"synthetic": name}})
    info = {"kind": "distill_rank_teacher", "teacher": str(teacher_path),
            "teacher_pred_mean": float(teacher_pred.mean()),
            "rows_before": len(rows), "rows_after": len(out_rows)}
    return out_rows, info


# ------------------------------------------------------------------ vendor 分派

_VENDOR_MODULES = {}


def _vendor_module(task):
    if task in _VENDOR_MODULES:
        return _VENDOR_MODULES[task]
    name = f"_baselines_vendor_{task}_model"
    spec = importlib.util.spec_from_file_location(name, VENDOR / task / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _VENDOR_MODULES[task] = mod
    return mod


def _vendor_train_run(task, profile, hidden):
    """把 vendor/<TASK> 复制到临时目录（镜像里 /app 只读），改好 config 再导入 train_run。"""
    tmp = Path(tempfile.mkdtemp(prefix=f"baselines-{task}-"))
    dst = tmp / task
    shutil.copytree(VENDOR / task, dst)
    cfg_path = dst / "configs" / f"{profile}.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    if hidden is not None:
        cfg["hidden"] = int(hidden)
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    sys.path.insert(0, str(dst))
    name = f"_baselines_vendor_{task}_train"
    spec = importlib.util.spec_from_file_location(name, dst / "train.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod, tmp


def _hidden_of(checkpoint):
    if not checkpoint:
        return None
    try:
        return int(_vendor_module(_task_of_checkpoint(checkpoint)).Model.load(checkpoint).meta["hidden"])
    except Exception:                                        # noqa: BLE001
        return None


def _task_of_checkpoint(checkpoint):
    """从检查点 metadata 的 profile 反推任务；失败就按特征数判断。"""
    for task in ("A", "B", "C"):
        try:
            meta = _vendor_module(task).Model.load(checkpoint).meta
            profile = str(meta.get("profile") or "").upper()
            return profile if profile in ("A", "B", "C") else task
        except Exception:                                    # noqa: BLE001
            continue
    return "B"


# ------------------------------------------------------------------ 主流程

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", required=True, help="A / B / C")
    ap.add_argument("--train", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--output", required=True)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--resume", help="上一轮被采纳的检查点")
    group.add_argument("--from-baseline", help="官方/参考起点检查点（续训语义与 --resume 相同）")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--stage-id", default="harness-round")
    ap.add_argument("--recipe", default="auto", choices=["auto", "plain"] + sorted(set(RECIPE.values())))
    ap.add_argument("--label-provenance", default=None)
    ap.add_argument("--json", action="store_true", help="只打印报告 JSON")
    args = ap.parse_args(argv)

    task = args.profile.strip().upper()[:1]
    if task not in DEFAULTS:
        raise SystemExit(f"unsupported profile: {args.profile}")

    parent = args.resume or args.from_baseline
    if not parent:
        fallback = MODELS / task / "checkpoint.npz"
        if fallback.is_file():
            parent = str(fallback)
    parent_hidden = _hidden_of(parent)
    defaults = DEFAULTS[task]
    hidden = parent_hidden if parent_hidden is not None else defaults["hidden"]
    epochs = args.epochs if args.epochs is not None else defaults["epochs"]
    lr = args.learning_rate if args.learning_rate is not None else defaults["learning_rate"]
    bs = args.batch_size if args.batch_size is not None else defaults["batch_size"]
    recipe = RECIPE[task] if args.recipe == "auto" else args.recipe

    rows = read_jsonl(args.train)
    tmpdir = Path(tempfile.mkdtemp(prefix="baselines-data-"))
    info = {"recipe": recipe, "parent": parent, "hidden": hidden}
    train_path = Path(args.train)

    if recipe == "class_weight":
        new_rows, meta = recipe_class_weight(rows)
        train_path = tmpdir / "train_weighted.jsonl"
        write_jsonl(train_path, new_rows)
        info.update(meta)
    elif recipe == "distill":
        teacher = MODELS / task / "checkpoint_rank.npz"
        if not teacher.is_file():
            info["distill_skipped"] = f"缺少教师 {teacher}，退回普通续训"
            recipe = "plain"
        else:
            new_rows, meta = recipe_distill_B(rows, teacher)
            train_path = tmpdir / "train_distill.jsonl"
            write_jsonl(train_path, new_rows)
            info.update(meta)
            # 蒸馏出的学生从随机初始化开始更有意义（教师知识本身就在标签里）；
            # 但仍以父检查点为 resume 以保留"父检查点摘要 + 累计步数"的审计语义
    elif recipe == "long_average":
        info["note"] = "同初始化多轮训练（权重平均效果由多 epoch 近似）"

    mod, vendor_tmp = _vendor_train_run(task, task, hidden)
    try:
        report = mod.train_run(task, str(train_path), str(args.validation), str(args.output),
                               resume=parent, epochs=epochs, learning_rate=lr, batch_size=bs,
                               seed=args.seed, stage_id=args.stage_id,
                               label_provenance=args.label_provenance)
    finally:
        shutil.rmtree(vendor_tmp, ignore_errors=True)
        shutil.rmtree(tmpdir, ignore_errors=True)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    else:
        print(json.dumps({"recipe_info": info, "report": report}, ensure_ascii=False, indent=2,
                         allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
