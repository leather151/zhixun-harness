"""A 包：批内秩变换的轮次/权重扫描（找它能不能稳定超过 0.727）。

上一轮发现：同样 3 个 epoch 从零训练，
  原始特征 0.4820（乙醛 recall 0.018）
  批内秩变换·不加权 0.6087（乙醛 recall 0.202）
  批内秩变换·类权   0.5707（乙醛 recall 0.394）
说明"批内秩变换"确实让乙醛从"完全看不见"变成"能看见一部分"。
但对照组训练预算不足（3 epoch from scratch vs 官方 20 epoch），不能直接比。

本脚本：
  1. 把三种特征（原始 / 批内秩 / 批内秩+类权）的轮次扫到 30，看收敛曲线；
  2. 同时报 Macro-F1 与乙醛 recall，判断"救乙醛"是否以牺牲其它类为代价；
  3. 结论只用于判断方向，不产出交付检查点（交付版另按 holdout 选轮次跑一次）。
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
PATHS = None


def batch_rank(x):
    out = np.empty_like(x, dtype=np.float64)
    n = len(x)
    for j in range(x.shape[1]):
        order = np.argsort(x[:, j], kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n, dtype=np.float64)
        out[:, j] = ranks / max(n - 1, 1)
    return out


def class_weights(y, n=6):
    counts = np.bincount(y.astype(int), minlength=n).astype(np.float64)
    counts[counts == 0] = 1
    return np.sqrt(len(y) / (n * counts))[y.astype(int)]


def metrics_of(pred, y):
    f1 = M.macro_f1(y.astype(int), pred, 6, observed_only=False)
    sel = y == ACE
    per = []
    for c in range(6):
        s = y == c
        rec = float(np.mean(pred[s] == c)) if s.sum() else 0.0
        tp = int(np.sum(s & (pred == c)))
        fp = int(np.sum(~s & (pred == c)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        per.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return f1, float(np.mean(pred == y)), float(np.mean(pred[sel] == ACE)), per


def sweep(tag, x_tr, y_tr, x_va, y_va, weighted, epochs_list):
    mods = T.load_vendor("A")
    model = mods["model"].Model("classification", PATHS, CLASSES, x_tr, y_tr,
                               hidden=32, seed=42, learning_rate=0.003, batch_size=128)
    w = class_weights(y_tr) if weighted else None
    rows = []
    done = 0
    for e in epochs_list:
        model.fit(x_tr, y_tr, e - done, sample_weight=w)
        done = e
        pred = model.predict(x_va).argmax(1)
        f1, acc, ace, per = metrics_of(pred, y_va)
        rows.append((e, f1, acc, ace, per))
        print(f"  [{tag}] epoch {e:2d}  Macro-F1 {f1:.4f}  Acc {acc:.4f}  "
              f"乙醛recall {ace:.3f}  逐类 " + " ".join(f"{v:.2f}" for v in per))
    return rows


def main():
    global PATHS
    train = T.load_jsonl(T.task_base("A") / "train.jsonl")
    valid = T.load_jsonl(T.task_base("A") / "validation.jsonl")
    PATHS = train["paths"]
    y_tr = np.asarray([CLASSES.index(v) for v in train["labels"]])
    y_va = np.asarray([CLASSES.index(v) for v in valid["labels"]])

    epochs_list = [3, 6, 10, 15, 20, 30]
    print("=" * 96)
    print("A 包：特征 × 轮次 扫描（各类顺序：" + "、".join(CN) + "）")
    print("=" * 96)
    print("参照：官方起点 0.7140 ｜ 当前定稿 0.7268（原始+类权+早停2轮，热启动）\n")

    x_tr, x_va = train["x"], valid["x"]
    r_raw = sweep("原始·不加权", x_tr, y_tr, x_va, y_va, False, epochs_list)
    print()
    r_raw_cw = sweep("原始·类权", x_tr, y_tr, x_va, y_va, True, epochs_list)
    print()
    xr_tr, xr_va = batch_rank(x_tr), batch_rank(x_va)
    r_rank = sweep("批内秩·不加权", xr_tr, y_tr, xr_va, y_va, False, epochs_list)
    print()
    r_rank_cw = sweep("批内秩·类权", xr_tr, y_tr, xr_va, y_va, True, epochs_list)

    print("\n" + "=" * 96)
    print("各自最优（按 Macro-F1）：")
    for name, rows in (("原始·不加权", r_raw), ("原始·类权", r_raw_cw),
                       ("批内秩·不加权", r_rank), ("批内秩·类权", r_rank_cw)):
        best = max(rows, key=lambda r: r[1])
        print(f"  {name:14s} 最优 epoch {best[0]:2d}：Macro-F1 {best[1]:.4f}  "
              f"乙醛 recall {best[3]:.3f}  逐类 " + " ".join(f"{v:.2f}" for v in best[4]))
    print("\n注意：上表的\"最优\"是按 validation 挑的，仅用于判断方向；"
          "交付版必须用 train 内部留出选轮次。")


if __name__ == "__main__":
    main()
