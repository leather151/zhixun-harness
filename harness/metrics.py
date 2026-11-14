"""指标与漂移度量（自包含、只用 numpy，镜像里可直接跑）。

与 03_实验/数据分析/common.py 的口径一致：
- 分类：macro-F1（对出现过的类取平均）、accuracy、混淆矩阵；
- 回归：RMSE、MAE、bias、NASA 非对称分（C 包）；
- 漂移：每特征两样本 KS 统计量、均值漂移（以参照集标准差为单位）、预测分布偏移。
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------- 基础指标

def rmse(pred, true):
    d = np.asarray(pred, float) - np.asarray(true, float)
    return float(np.sqrt(np.mean(d ** 2)))


def mae(pred, true):
    d = np.asarray(pred, float) - np.asarray(true, float)
    return float(np.mean(np.abs(d)))


def bias(pred, true):
    d = np.asarray(pred, float) - np.asarray(true, float)
    return float(np.mean(d))


def nasa_score(pred, true):
    """NASA C-MAPSS 非对称评分：高估惩罚更重（逐样本罚分之和，越低越好）。"""
    d = np.asarray(pred, float) - np.asarray(true, float)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def macro_f1(true, pred, n_classes, observed_only=True):
    """Macro-F1。

    `observed_only=True` 与官方 `baselines/train.py` 一致：只在"真实或预测出现过的类"上取平均
    （否则一个只含单一类别的窗口会被压到 1/6，失去监控意义）。
    `observed_only=False` 时对固定的 n_classes 个类平均 —— 整集比较（0.7140 那个口径）用这个。
    """
    true = np.asarray(true)
    pred = np.asarray(pred)
    classes = (sorted(set(true.tolist()) | set(pred.tolist()))
               if observed_only else list(range(n_classes)))
    f1s = []
    for c in classes:
        tp = int(np.sum((true == c) & (pred == c)))
        fp = int(np.sum((true != c) & (pred == c)))
        fn = int(np.sum((true == c) & (pred != c)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def confusion(true, pred, n_classes):
    mat = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(true, pred):
        mat[int(t), int(p)] += 1
    return mat


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")

    def rank(x):
        order = np.argsort(x, kind="mergesort")
        r = np.empty(len(x))
        sx = x[order]
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and sx[j + 1] == sx[i]:
                j += 1
            r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
            i = j + 1
        return r

    ra, rb = rank(a[ok]), rank(b[ok])
    ra -= ra.mean()
    rb -= rb.mean()
    den = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den else float("nan")


# ---------------------------------------------------------------- 漂移度量

def ks_stat(a, b):
    a = np.sort(np.asarray(a, float))
    b = np.sort(np.asarray(b, float))
    if len(a) < 2 or len(b) < 2:
        return 0.0
    grid = np.concatenate([a, b])
    fa = np.searchsorted(a, grid, side="right") / len(a)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def drift_report(ref_x, new_x, ref_pred=None, new_pred=None, feature_names=None):
    """返回输入漂移 + 预测漂移的可解释摘要。"""
    ref_x = np.asarray(ref_x, float)
    new_x = np.asarray(new_x, float)
    d = ref_x.shape[1]
    sd = ref_x.std(0)
    sd[sd < 1e-12] = 1
    z = (new_x.mean(0) - ref_x.mean(0)) / sd
    ks = np.array([ks_stat(ref_x[:, j], new_x[:, j]) for j in range(d)])
    order = np.argsort(-np.abs(z))[:5]
    names = feature_names or [f"f{j + 1}" for j in range(d)]
    out = {
        "ks_median": float(np.median(ks)),
        "ks_max": float(ks.max()),
        "ks_over_0_2": int((ks > 0.2).sum()),
        "abs_z_median": float(np.median(np.abs(z))),
        "abs_z_max": float(np.max(np.abs(z))),
        "top_shifted": [{"feature": names[j], "z": float(z[j]), "ks": float(ks[j])} for j in order
                        if abs(z[j]) > 1e-9],
    }
    if ref_pred is not None and new_pred is not None:
        ref_pred = np.asarray(ref_pred, float)
        new_pred = np.asarray(new_pred, float)
        spread = float(ref_pred.std()) or 1.0
        out["pred_mean_shift_in_sigma"] = float((new_pred.mean() - ref_pred.mean()) / spread)
        out["pred_std_ratio"] = float((new_pred.std() or 1e-12) / spread)
    return out
