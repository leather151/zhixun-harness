"""B 包第二轮蒸馏：在输入域内"无限查询教师"，让学生学的是整个定义域。

第一轮（distill_b_raw.py）只在训练流形上监督：teacher 的预测只在 7032 行真实样本上取过，
留出集 RMSE 194 很好看，但换到官方 validation（不同轴承、原始特征落在流形外）就掉到 584
（pearson −0.08）——典型的"分布内拟合、分布外外推失败"。

关键洞察：教师（秩模型）是一个**确定的纯函数**，它接受任意原始特征向量（先把每维映射成
分位数再前向）。所以可以：
  1. 在每维训练观测范围内（两边各外扩 15%）均匀采样；
  2. 再把真实样本两两线性插值（覆盖特征间相关性）；
  3. 用教师给这些"虚拟输入"打标签；
  4. 学生只吃原始特征去拟合教师 —— 覆盖整个定义域，而不是只覆盖训练流形。

合规：
- 教师已在官方 train.jsonl 上拟合；虚拟输入由训练集统计量决定，**不碰 validation**；
- 选型仍用 train 内部按轴承留出（Bearing2_3 / Bearing3_3）；
- 官方 validation 只在定稿后测一次。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP / "harness"))

import preprocess as P       # noqa: E402
import tasks as T            # noqa: E402

HOLDOUT_GROUPS = T.TASKS["B"]["holdout"]["groups"]
RNG = np.random.default_rng(20261115)


def metrics(pred, y):
    e = pred - y
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)), "pearson": float(np.corrcoef(pred, y)[0, 1])}


def bearing_weight(groups):
    uniq, counts = np.unique(groups, return_counts=True)
    per = {g: 1.0 / c for g, c in zip(uniq, counts)}
    w = np.array([per[g] for g in groups], dtype=np.float64)
    return w / w.mean()


def synth_inputs(x, n_uniform, n_mix, pad=0.15):
    """虚拟输入：盒子内均匀采样 + 真实样本两两插值。"""
    lo, hi = x.min(0), x.max(0)
    span = hi - lo
    lo = lo - pad * span
    hi = hi + pad * span
    u = RNG.uniform(lo, hi, size=(n_uniform, x.shape[1]))
    i = RNG.integers(0, len(x), size=n_mix)
    j = RNG.integers(0, len(x), size=n_mix)
    t = RNG.uniform(0.0, 1.0, size=(n_mix, 1))
    mix = x[i] * t + x[j] * (1 - t)
    return np.vstack([x, u, mix])          # 真实样本也一起喂（保底）


def main():
    mod = T.load_vendor("B")["model"]
    base = T.task_base("B")
    train = T.load_jsonl(base / "train.jsonl")
    valid = T.load_jsonl(base / "validation.jsonl")
    groups = np.array([str(m.get("bearing", "?")) for m in train["meta"]], dtype=object)
    print(f"[data] train {len(train['y'])} 行 / {len(set(groups))} 轴承；valid {len(valid['y'])} 行")

    teacher = mod.Model.load(T.checkpoint_path("B", "checkpoint_rank.npz"))
    t_valid = teacher.predict(P.transform_inputs(teacher.meta, valid["x"]))
    print("[teacher] validation(测量)", {k: round(v, 2) for k, v in metrics(t_valid, valid["y"]).items()})

    # ---- 构造监督集 ----
    xs = synth_inputs(train["x"], n_uniform=150_000, n_mix=80_000)
    ys = teacher.predict(P.transform_inputs(teacher.meta, xs))
    w = np.concatenate([bearing_weight(groups), np.full(len(xs) - len(train["y"]), 0.5)])
    keep = RNG.random(len(xs)) < 0.85          # 抽样控制内存
    xs, ys, w = xs[keep], ys[keep], w[keep]
    print(f"[synth] 监督集 {len(xs)} 行（真实 {keep[:len(train['y'])].sum()} + 虚拟）"
          f"；教师预测范围 [{ys.min():.0f}, {ys.max():.0f}]")

    # ---- 选型：train 内部按轴承留出 ----
    is_hold = np.array([g in HOLDOUT_GROUPS for g in groups])
    ho_idx = np.where(is_hold)[0]
    best = None
    print("\n[search] hidden × epochs（合成监督，按轴承等权，目标=教师）")
    for hidden in (96, 128):
        for epochs in (4, 8, 14):
            for wd in (0.0, 1e-3):
                m = mod.Model("regression", train["paths"], [], train["x"], ys[: len(train["x"])],
                              hidden=hidden, seed=42, learning_rate=3e-3, batch_size=256)
                m.fit(xs, ys, epochs=epochs, sample_weight=w, weight_decay=wd)
                ho = metrics(m.predict(train["x"][ho_idx]), train["y"][ho_idx])
                gap = metrics(m.predict(train["x"][ho_idx]),
                              teacher.predict(P.transform_inputs(teacher.meta, train["x"][ho_idx])))
                print(f"  hidden={hidden:3d} epochs={epochs:2d} wd={wd:g}  "
                      f"留出 RMSE={ho['rmse']:7.2f} MAE={ho['mae']:7.2f} bias={ho['bias']:+7.2f} "
                      f"ρ={ho['pearson']:+.3f}  |对教师|={gap['rmse']:7.2f}")
                if best is None or ho["rmse"] < best["holdout"]["rmse"]:
                    best = {"hidden": hidden, "epochs": epochs, "wd": wd,
                            "holdout": ho, "gap": gap}
    print(f"\n[best] hidden={best['hidden']} epochs={best['epochs']} wd={best['wd']}  "
          f"留出 RMSE={best['holdout']['rmse']:.2f} ρ={best['holdout']['pearson']:+.3f}")

    # ---- 定稿 ----
    final = mod.Model("regression", train["paths"], [], train["x"], ys[: len(train["x"])],
                      hidden=best["hidden"], seed=42, learning_rate=3e-3, batch_size=256)
    final.fit(xs, ys, epochs=best["epochs"], sample_weight=w, weight_decay=best["wd"])
    final.meta["profile"] = "B"
    final.meta["target_unit"] = "minutes"
    final.meta["preprocess_note"] = (
        f"raw 特征（平台无需外部预处理）· 蒸馏自秩模型 · hidden={best['hidden']} "
        f"epochs={best['epochs']} wd={best['wd']:g} · 合成域内监督 + 按轴承等权")
    final.meta["distill_from"] = "models/B/checkpoint_rank.npz"
    final.meta["distill_supervision"] = int(len(xs))

    val = metrics(final.predict(valid["x"]), valid["y"])
    print("[定稿] 官方 validation（只测一次）:", {k: round(v, 2) for k, v in val.items()})

    prev = APP / "models" / "B" / "checkpoint_raw_distilled.npz"
    if prev.is_file():
        print("[对照] 第一轮(流形蒸馏):",
              {k: round(v, 2) for k, v in metrics(mod.Model.load(prev).predict(valid["x"]),
                                                  valid["y"]).items()})
    bl = mod.Model.load(T.checkpoint_path("B", "checkpoint.npz"))
    print("[对照] 官方基线:", {k: round(v, 2) for k, v in metrics(bl.predict(valid["x"]),
                                                                 valid["y"]).items()})

    out = APP / "models" / "B" / "checkpoint_raw_distilled2.npz"
    if out.exists():
        out.unlink()
    final.save(out)
    print(f"[save] {out} ({out.stat().st_size} bytes)")
    (APP / "runs" / "B_distill2_report.json").write_text(
        json.dumps({"teacher_validation": metrics(t_valid, valid["y"]),
                    "student_validation": val, "selected": best,
                    "supervision_rows": int(len(xs)),
                    "checkpoint": str(out.relative_to(APP))},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
