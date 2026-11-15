"""A 包诊断：乙醛（Acetaldehyde）到底能不能与乙醇（Ethanol）分开？

背景：官方检查点在验证集(b02)上把 109 条乙醛**全部**判错，且我们的类别加权/秩变换
也没能救回来（乙醛 recall 恒为 0）。两种可能：
  (a) 信息存在，但 b01→b02 的漂移让决策边界失效（可救：需要域自适应/对齐）；
  (b) 在 b02 里这两类本来就高度重叠（不可救：Macro-F1 天花板就在 0.72 附近）。

本脚本用**批内交叉验证**判定：
  - 在 b01 内部做 Ethanol vs Acetaldehyde 二分类（同分布，看信息量上限）；
  - 在 b02 内部做同样的二分类（跨批后信息还剩多少）。
只用于诊断结论，**不参与任何模型训练**。
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
PAIR = ["Ethanol", "Acetaldehyde"]     # 0 = 乙醇, 1 = 乙醛


def pair_data(rows):
    y = np.asarray([CLASSES.index(v) for v in rows["labels"]])
    sel = np.isin(y, [CLASSES.index(c) for c in PAIR])
    return rows["x"][sel], (y[sel] == CLASSES.index(PAIR[1])).astype(int), sel


def knn_cv(x, y, k=1, folds=5, seed=0):
    """简单 K 折交叉验证的 kNN 准确率（欧氏、标准化）。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    accs = []
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        mu, sd = x[train].mean(0), x[train].std(0)
        sd[sd < 1e-12] = 1
        ztr, zte = (x[train] - mu) / sd, (x[test] - mu) / sd
        pred = M.__dict__.get("_knn")
        # 暴力 kNN（样本量小，直接算）
        d = ((zte[:, None, :] - ztr[None, :, :]) ** 2).sum(2)
        nn = np.argpartition(d, k - 1, axis=1)[:, :k]
        votes = y[train][nn]
        out = (votes.mean(axis=1) >= 0.5).astype(int)
        accs.append(float(np.mean(out == y[test])))
    return float(np.mean(accs))


def lda_cv(x, y, folds=5, seed=0, reg=1e-2):
    """闭式 LDA（线性判别）交叉验证：比 kNN 更能代表"线性可分性"。"""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    accs = []
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        mu, sd = x[train].mean(0), x[train].std(0)
        sd[sd < 1e-12] = 1
        ztr, zte = (x[train] - mu) / sd, (x[test] - mu) / sd
        m0, m1 = ztr[y[train] == 0].mean(0), ztr[y[train] == 1].mean(0)
        Sw = np.cov(ztr[y[train] == 0].T) + np.cov(ztr[y[train] == 1].T)
        Sw = Sw + reg * np.eye(Sw.shape[0])
        w = np.linalg.solve(Sw, m1 - m0)
        b = -0.5 * (m0 + m1) @ w
        out = (zte @ w + b > 0).astype(int)
        accs.append(float(np.mean(out == y[test])))
    return float(np.mean(accs))


def main():
    train = T.load_jsonl(T.task_base("A") / "train.jsonl")
    valid = T.load_jsonl(T.task_base("A") / "validation.jsonl")

    print("=" * 74)
    print("A 包诊断：乙醇 vs 乙醛 的批内可分性（只做诊断，不训练任何交付模型）")
    print("=" * 74)

    for tag, rows in (("b01（训练批）", train), ("b02（验证批）", valid)):
        x, y, _ = pair_data(rows)
        n0, n1 = int((y == 0).sum()), int((y == 1).sum())
        print(f"\n[{tag}] 乙醇 {n0} 条 / 乙醛 {n1} 条，特征 {x.shape[1]} 维")
        if n0 < 5 or n1 < 5:
            print("  样本太少，跳过")
            continue
        print(f"  1-NN（批内 5 折）：   {knn_cv(x, y, k=1):.3f}")
        print(f"  5-NN（批内 5 折）：   {knn_cv(x, y, k=5):.3f}")
        print(f"  LDA（批内 5 折）：    {lda_cv(x, y):.3f}")
        # 两类在标准化空间的中心距离（以类内标准差为单位）
        mu, sd = x.mean(0), x.std(0)
        sd[sd < 1e-12] = 1
        z = (x - mu) / sd
        d = np.linalg.norm(z[y == 0].mean(0) - z[y == 1].mean(0)) / np.sqrt(z.shape[1])
        print(f"  两类质心的标准化距离：{d:.3f}（越小越重叠）")

    # 跨批错位程度：b01 的乙醛中心 vs b02 的乙醛中心 / b02 的乙醇中心
    xt, yt, _ = pair_data(train)
    xv, yv, _ = pair_data(valid)
    mu, sd = xt.mean(0), xt.std(0)
    sd[sd < 1e-12] = 1
    t_eth, t_ace = (xt[yt == 0] - mu) / sd, (xt[yt == 1] - mu) / sd
    v_eth, v_ace = (xv[yv == 0] - mu) / sd, (xv[yv == 1] - mu) / sd

    def dist(a, b):
        return float(np.linalg.norm(a.mean(0) - b.mean(0)) / np.sqrt(a.shape[1]))

    print("\n跨批的中心位移（用 b01 的标准化尺度，除以 sqrt(维数)）：")
    print(f"  b01乙醛 → b02乙醛 : {dist(t_ace, v_ace):.3f}   （同类跨批漂移）")
    print(f"  b01乙醛 → b02乙醇 : {dist(t_ace, v_eth):.3f}   （漂到别的类去了？）")
    print(f"  b01乙醛 → b01乙醇 : {dist(t_ace, t_eth):.3f}   （批内两类距离，作为参照）")
    print(f"  b02乙醛 → b02乙醇 : {dist(v_ace, v_eth):.3f}   （b02 内部两类距离）")

    print("\n判读：")
    print("  · 若 b02 批内 LDA/1-NN 明显 >0.5（例如 >0.85）→ 信息存在，问题在“跨批对齐”，")
    print("    值得做域自适应（CORAL/最优传输/按批标准化重估）。")
    print("  · 若 b02 批内也接近 0.5 → 两类在 b02 里本来就重叠，Macro-F1 天花板约 0.72，")
    print("    应停止在乙醛上做文章，把精力放到别的组。")


if __name__ == "__main__":
    main()
