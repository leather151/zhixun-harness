"""A 包：按 Fisher 判别比选特征（最后一试）。

与之前"去漂移特征"不同：那个按**跨时间稳定性**删；这个按**类间可分性**留。
动机：445 条训练样本 / 128 维特征，噪声维很多；只保留判别力最强的前 k 维可能提升泛化。

同时报：
  - 用 b01 全部样本算 Fisher（选择统计量只用训练集）；
  - 轮次固定 20（与"原始·类权"收敛点一致），只在官方 validation 报一次；
  - 由于 k 是扫描出来的，本脚本只用于判断方向，不作为交付依据。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import metrics as M      # noqa: E402
import tasks as T        # noqa: E402

CLASSES = T.TASKS["A"]["classes"]
CN = T.TASKS["A"]["cn"]
ACE = CLASSES.index("Acetaldehyde")


def fisher_ratio(x, y, n_classes=6):
    overall = x.mean(0)
    within = np.zeros(x.shape[1])
    between = np.zeros(x.shape[1])
    for c in range(n_classes):
        sel = y == c
        if not sel.any():
            continue
        xc = x[sel]
        within += ((xc - xc.mean(0)) ** 2).sum(0)
        between += ((xc.mean(0) - overall) ** 2) * sel.sum()
    return between / (within + 1e-30)


def class_weights(y, n=6):
    counts = np.bincount(y.astype(int), minlength=n).astype(np.float64)
    counts[counts == 0] = 1
    return np.sqrt(len(y) / (n * counts))[y.astype(int)]


def run(x_tr, y_tr, x_va, y_va, name, epochs=20):
    mods = T.load_vendor("A")
    paths = [PATHS[j] for j in range(x_tr.shape[1])]
    model = mods["model"].Model("classification", paths, CLASSES, x_tr, y_tr,
                               hidden=32, seed=42, learning_rate=0.003, batch_size=128)
    model.fit(x_tr, y_tr, epochs, sample_weight=class_weights(y_tr))
    pred = model.predict(x_va).argmax(1)
    f1 = M.macro_f1(y_va.astype(int), pred, 6, observed_only=False)
    per = []
    for c in range(6):
        s = y_va == c
        rec = float(np.mean(pred[s] == c)) if s.any() else 0.0
        tp = int(np.sum(s & (pred == c)))
        fp = int(np.sum(~s & (pred == c)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        per.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    sel = y_va == ACE
    return {"name": name, "f1": f1, "acc": float(np.mean(pred == y_va)),
            "ace": float(np.mean(pred[sel] == ACE)) if sel.any() else 0.0, "per": per}


PATHS = None


def main():
    global PATHS
    train = T.load_jsonl(T.task_base("A") / "train.jsonl")
    valid = T.load_jsonl(T.task_base("A") / "validation.jsonl")
    PATHS = train["paths"]
    y_tr = np.asarray([CLASSES.index(v) for v in train["labels"]])
    y_va = np.asarray([CLASSES.index(v) for v in valid["labels"]])
    x_tr, x_va = train["x"], valid["x"]

    fr = fisher_ratio(x_tr, y_tr)
    order = np.argsort(-fr)

    print("=" * 92)
    print("A 包：Fisher 判别比选特征（k 维）·轮次固定 20（各类顺序：" + "、".join(CN) + "）")
    print("=" * 92)
    print("参照：官方 0.7140 ｜ 当前定稿 0.7268 ｜ 前 15 强特征："
          + ", ".join(f"F{j + 1}" for j in order[:15]) + "\n")

    rows = [run(x_tr, y_tr, x_va, y_va, f"全部 {x_tr.shape[1]} 维（对照）")]
    for k in (16, 32, 48, 64, 96):
        idx = order[:k]
        rows.append(run(x_tr[:, idx], y_tr, x_va[:, idx], y_va, f"Fisher 前 {k} 维"))

    print(f"{'方案':22s} {'Macro-F1':>9s} {'Acc':>7s} {'乙醛recall':>10s}   逐类F1")
    print("-" * 92)
    for r in rows:
        print(f"{r['name']:22s} {r['f1']:9.4f} {r['acc']:7.4f} {r['ace']:10.3f}   "
              + " ".join(f"{v:.2f}" for v in r["per"]))
    best = max(rows, key=lambda r: r["f1"])
    print(f"\n最优：{best['name']}  Macro-F1 {best['f1']:.4f}")
    print("判读：只有明显超过 0.7268（且乙醛 recall 起来）才值得继续；否则 A 到此为止。")


if __name__ == "__main__":
    main()
