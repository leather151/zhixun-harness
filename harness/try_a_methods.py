"""A 包：换三种思路再试一次（都只用无标签的批量统计或训练集统计）。

已失败的方法（见 实验日志 §4）：
  - 类别加权：乙醛 recall 仍 0（加权不能让模型"看见"）；
  - 秩变换（用 b01 的 ECDF 映射 b02）：0.6942，更差；
  - 剔除漂移特征：乙醛救回来了（0.982）但其它类崩，总分 0.6620。

本脚本试三种**思路不同**的方法：

  M1 批内秩变换（batch-wise rank）
     假设：跨批的"绝对水平"不可比，但**批内相对位置**可比。
     做法：训练时把 b01 的每列换成"在本批内的分位"，推理时把 b02 的每列换成"在 b02 内的分位"。
     两批的边缘分布被强制变成同一个均匀分布 → 分布对齐是"构造出来"的，不靠估计。
     代价：需要一批样本才能算分位（单条流式推理不适用），必须写明。

  M2 CORAL 对齐（二阶统计量对齐，用 b02 的**无标签**输入）
     假设：批间差异主要是协方差/均值漂移，可以用线性变换搬回去。
     做法：Xb02' = (Xb02 − μt)·Ct^(−1/2)·Cs^(1/2) + μs，然后喂给"在 b01 上训练"的模型。
     与 EDA 里失败的"逐特征 z-score"不同：CORAL 处理的是**特征间的相关性结构**。

  M3 稳健缩放（median / IQR 代替 mean / std）
     假设：少数重尾特征（如 F42，跨批 11.5σ）把标准化拉爆，tanh 单元被推到饱和。
     做法：每特征用中位数与四分位距缩放，重新训练。

统一协议：轮次固定 3（a priori，不做 validation 搜索），只在官方 validation 上报一次。
只用于判断"这条路值不值得继续"，不产出交付检查点。
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
ACETALDEHYDE = CLASSES.index("Acetaldehyde")
EPOCHS = 3


# ------------------------------------------------------------------ 变换

def batch_rank(x):
    """每列换成"在本批内的分位"（0–1）。"""
    out = np.empty_like(x, dtype=np.float64)
    n = len(x)
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        out[:, j] = ranks / max(n - 1, 1)
    return out


def robust_scale(x_ref, x):
    """用参照集的中位数/IQR 缩放。"""
    med = np.median(x_ref, axis=0)
    q75, q25 = np.percentile(x_ref, [75, 25], axis=0)
    iqr = q75 - q25
    iqr[iqr < 1e-12] = 1.0
    return (x - med) / iqr


def coral(x_src, x_tgt):
    """把 x_tgt 的二阶统计量对齐到 x_src（均为无标签）。"""
    def cov(x):
        xc = x - x.mean(0)
        return np.cov(xc, rowvar=False) + 1e-6 * np.eye(x.shape[1])

    Cs, Ct = cov(x_src), cov(x_tgt)
    # 对称矩阵的逆平方根/平方根
    def mat_pow(m, p):
        w, v = np.linalg.eigh(m)
        w = np.clip(w, 1e-8, None)
        return v @ np.diag(w ** p) @ v.T

    xt_c = x_tgt - x_tgt.mean(0)
    return (xt_c @ mat_pow(Ct, -0.5)) @ mat_pow(Cs, 0.5) + x_src.mean(0)


# ------------------------------------------------------------------ 训练

def class_weights(y, n=6):
    counts = np.bincount(y.astype(int), minlength=n).astype(np.float64)
    counts[counts == 0] = 1
    return np.sqrt(len(y) / (n * counts))[y.astype(int)]


def train_eval(x_tr, y_tr, x_va, y_va, name, weighted=True, epochs=EPOCHS):
    mods = T.load_vendor("A")
    model = mods["model"].Model("classification", PATHS, CLASSES, x_tr, y_tr,
                               hidden=32, seed=42, learning_rate=0.003, batch_size=128)
    w = class_weights(y_tr) if weighted else None
    model.fit(x_tr, y_tr, epochs, sample_weight=w)
    pred = model.predict(x_va).argmax(1)
    f1 = M.macro_f1(y_va.astype(int), pred, 6, observed_only=False)
    acc = float(np.mean(pred == y_va))
    sel = y_va == ACETALDEHYDE
    ace_recall = float(np.mean(pred[sel] == ACETALDEHYDE)) if sel.sum() else float("nan")
    per_class = []
    for c in range(6):
        s = y_va == c
        rec = float(np.mean(pred[s] == c)) if s.sum() else 0.0
        tp = int(np.sum(s & (pred == c)))
        fp = int(np.sum(~s & (pred == c)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        per_class.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return {"name": name, "macro_f1": f1, "accuracy": acc,
            "acetaldehyde_recall": ace_recall, "per_class_f1": per_class}


PATHS = None


def main():
    global PATHS
    train = T.load_jsonl(T.task_base("A") / "train.jsonl")
    valid = T.load_jsonl(T.task_base("A") / "validation.jsonl")
    PATHS = train["paths"]
    y_tr = np.asarray([CLASSES.index(v) for v in train["labels"]])
    y_va = np.asarray([CLASSES.index(v) for v in valid["labels"]])
    x_tr, x_va = train["x"], valid["x"]

    print("=" * 78)
    print("A 包：三种新思路（轮次固定 3，只在官方 validation 报一次，不做搜索）")
    print("=" * 78)
    print("参照：官方起点 Macro-F1 0.7140 ｜ 当前定稿 0.7268（类别加权+早停）")
    print("     已失败：秩变换(ECDF) 0.6942 ｜ 去漂移特征 0.6620\n")

    results = []

    # ---- M0 对照：原始特征 + 类权（已知 ~0.72）----
    results.append(train_eval(x_tr, y_tr, x_va, y_va, "M0 原始特征·类权（对照）"))

    # ---- M1 批内秩变换 ----
    results.append(train_eval(batch_rank(x_tr), y_tr, batch_rank(x_va), y_va,
                              "M1 批内秩变换·类权"))
    results.append(train_eval(batch_rank(x_tr), y_tr, batch_rank(x_va), y_va,
                              "M1' 批内秩变换·不加权", weighted=False))

    # ---- M2 CORAL 对齐（b02 无标签）----
    results.append(train_eval(x_tr, y_tr, coral(x_tr, x_va), y_va,
                              "M2 CORAL对齐·类权"))
    # 同时看官方检查点在 CORAL 后的表现（不重训）
    mods = T.load_vendor("A")
    base = mods["model"].Model.load(T.MODEL_ROOT / "A" / "checkpoint.npz")
    pred = base.predict(coral(x_tr, x_va)).argmax(1)
    f1 = M.macro_f1(y_va.astype(int), pred, 6, observed_only=False)
    sel = y_va == ACETALDEHYDE
    results.append({"name": "M2' 官方检查点 + CORAL（不重训）", "macro_f1": f1,
                    "accuracy": float(np.mean(pred == y_va)),
                    "acetaldehyde_recall": float(np.mean(pred[sel] == ACETALDEHYDE)),
                    "per_class_f1": [0.0] * 6})

    # ---- M3 稳健缩放 ----
    results.append(train_eval(robust_scale(x_tr, x_tr), y_tr,
                              robust_scale(x_tr, x_va), y_va,
                              "M3 稳健缩放(median/IQR)·类权"))

    # ---- 汇总 ----
    print(f"{'方法':30s} {'Macro-F1':>9s} {'Acc':>7s} {'乙醛recall':>10s}   逐类F1")
    print("-" * 78)
    for r in results:
        pc = " ".join(f"{v:.2f}" for v in r["per_class_f1"])
        print(f"{r['name']:30s} {r['macro_f1']:9.4f} {r['accuracy']:7.4f} "
              f"{r['acetaldehyde_recall']:10.3f}   {pc}")

    best = max(results, key=lambda r: r["macro_f1"])
    print("\n各类别顺序：" + "、".join(CN))
    print(f"\n最优：{best['name']}  Macro-F1 {best['macro_f1']:.4f}"
          f"（相对官方起点 {(best['macro_f1'] / 0.7140 - 1) * 100:+.1f}%）")
    print("判读：只有当某个方法把**乙醛 recall** 拉起来、同时其它类不明显下滑时，"
          "才说明找到真路子；否则仍是噪声内浮动。")
    return results


if __name__ == "__main__":
    main()
