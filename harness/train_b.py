"""B 包模型优化实验：秩变换 × 按轴承等权 × 正则（配置搜索 + 定稿）

设计依据：`03_实验/数据分析/B_分析.md`
  - 官方起点检查点 RMSE 681.16，连"常数预测（训练集中位数）"470.9 都不如，
    预测与真实 RUL 的秩相关是 −0.327（方向反了）；
  - 岭回归探针：原始 501 / 秩+不加权 529（方向对、尺度错）/ 秩+按轴承等权 **293**
    → 秩变换与按组等权缺一不可，且**收益高度依赖正则强度**。

本脚本做四件事：
  1. 在 **train 内部留出的两根轴承**上选轮次（不使用官方 validation 选配置）；
  2. 搜一组候选（初始化方式 × 权重衰减），全部用秩特征 + 按轴承等权；
  3. 用选定轮次在**全部训练轴承**上重训，官方 validation **只报一次**；
  4. 产出可直接部署的检查点 `models/B/checkpoint_rank.npz`（内含秩分位表）。

合规：
  - 分位表 / 标准化 / 目标缩放只在官方 train.jsonl 上拟合；
  - 选配置与选轮次只用 train 内部留出轴承（Bearing2_3 / Bearing3_3）；
  - 权重初始化自官方起点检查点时会显式声明为"热启动"，不是纯续训。

用法：
    python harness/train_b.py                  # 完整搜索 + 定稿
    python harness/train_b.py --quick          # 只跑历史最优配置
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import preprocess as P     # noqa: E402
import tasks as T          # noqa: E402

HOLDOUT_BEARINGS = ("Bearing2_3", "Bearing3_3")   # 仅用于选轮次
LR = 0.003
BATCH = 128
HIDDEN = 24

RANK_TABLE = None
RANKED = False


# ------------------------------------------------------------------ 工具

def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def mae(p, y):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def per_bearing(pred, y, bearings):
    out = {}
    for b in sorted(set(bearings.tolist())):
        sel = bearings == b
        out[b] = {"n": int(sel.sum()), "rmse": rmse(pred[sel], y[sel]),
                  "mae": mae(pred[sel], y[sel]),
                  "bias": float(np.mean(pred[sel] - y[sel]))}
    return out


def bearing_weights(bearings):
    counts = {b: int(np.sum(bearings == b)) for b in set(bearings.tolist())}
    w = np.array([1.0 / counts[b] for b in bearings])
    return w / w.mean()


def ridge_reference(x_tr, y_tr, w, x_va, y_va, lam=100.0):
    """秩特征 + 按轴承等权的岭回归，作为"线性可达到"的参照。"""
    mu, sd = x_tr.mean(0), x_tr.std(0)
    sd[sd < 1e-12] = 1
    Z = np.hstack([(x_tr - mu) / sd, np.ones((len(x_tr), 1))])
    Zv = np.hstack([(x_va - mu) / sd, np.ones((len(x_va), 1))])
    ym = float(np.average(y_tr, weights=w))
    reg = lam * np.diag(np.r_[np.ones(Z.shape[1] - 1), 0.0])
    W = np.linalg.solve((Z * w[:, None]).T @ Z + reg, (Z * w[:, None]).T @ (y_tr - ym))
    pred = np.maximum(Zv @ W + ym, 0.0)
    return {"rmse": rmse(pred, y_va), "mae": mae(pred, y_va),
            "bias": float(np.mean(pred - y_va))}


# ------------------------------------------------------------------ 训练

def build_model(x_fit, y_fit, paths, init_from):
    mods = T.load_vendor("B")
    model = mods["model"].Model("regression", paths, [], x_fit, y_fit,
                               hidden=HIDDEN, seed=42, learning_rate=LR, batch_size=BATCH)
    if init_from:
        base = mods["model"].Model.load(init_from)
        if base.params["w1"].shape != model.params["w1"].shape:
            raise ValueError("权重形状不一致，无法热启动")
        for key in model.params:
            model.params[key] = base.params[key].copy()
        model.meta["init_from"] = str(init_from)
    else:
        model.meta["init_from"] = "random"
    # 记录来源，便于审计
    model.meta["recipe"] = {"rank": RANKED, "init": model.meta["init_from"]}
    return model


def select_epochs(name, x_tr, y_tr, b_tr, weights, paths, init_from, wd, max_epochs):
    """在 train 内部留出轴承上选轮次（不使用官方 validation）。"""
    mask = ~np.isin(b_tr, HOLDOUT_BEARINGS)
    fit_idx, hold_idx = np.where(mask)[0], np.where(~mask)[0]
    model = build_model(x_tr[fit_idx], y_tr[fit_idx], paths, init_from)
    curve = []
    for epoch in range(1, max_epochs + 1):
        model.fit(x_tr[fit_idx], y_tr[fit_idx], 1,
                  sample_weight=weights[fit_idx] if weights is not None else None,
                  weight_decay=wd)
        curve.append({"epoch": epoch,
                      "holdout_rmse": rmse(model.predict(x_tr[hold_idx]), y_tr[hold_idx])})
    best = min(curve, key=lambda r: r["holdout_rmse"])
    return {"name": name, "curve": curve, "best_epoch": best["epoch"],
            "holdout_rmse": best["holdout_rmse"]}


def finalize(name, x_tr, y_tr, b_tr, x_va, y_va, b_va, weights, paths, init_from, wd, epochs):
    model = build_model(x_tr, y_tr, paths, init_from)
    model.fit(x_tr, y_tr, epochs, sample_weight=weights, weight_decay=wd)
    if RANKED:
        model.meta["preprocess"] = P.table_to_meta(RANK_TABLE)
    pred = model.predict(x_va)
    return {"name": name, "epochs": epochs, "model": model, "pred": pred,
            "rmse": rmse(pred, y_va), "mae": mae(pred, y_va),
            "bias": float(np.mean(pred - y_va)),
            "spearman": float(__import__("metrics").spearman(pred, y_va)),
            "per_bearing": per_bearing(pred, y_va, b_va)}


# ------------------------------------------------------------------ 主流程

def main(argv=None):
    global RANK_TABLE, RANKED
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="只跑历史最优配置")
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--init", default=str(T.MODEL_ROOT / "B" / "checkpoint.npz"))
    args = ap.parse_args(argv)

    base_dir = T.task_base("B")
    train = T.load_jsonl(base_dir / "train.jsonl")
    valid = T.load_jsonl(base_dir / "validation.jsonl")
    paths = train["paths"]
    b_tr = np.array([m["bearing"] for m in train["meta"]], dtype=object)
    b_va = np.array([m["bearing"] for m in valid["meta"]], dtype=object)
    y_tr, y_va = train["y"], valid["y"]

    table = P.fit_rank_table(train["x"])
    x_rank_tr, x_rank_va = P.apply_rank(table, train["x"]), P.apply_rank(table, valid["x"])
    w = bearing_weights(b_tr)

    # ---------- 参照：岭回归 ----------
    ridge_rank = ridge_reference(x_rank_tr, y_tr, w, x_rank_va, y_va)
    ridge_raw = ridge_reference(train["x"], y_tr, w, valid["x"], y_va)

    print("=" * 80)
    print("B 包模型优化：秩变换 × 按轴承等权 × 正则")
    print("=" * 80)
    print(f"train {len(y_tr)} 行 / {len(set(b_tr.tolist()))} 根轴承；选轮次留出 "
          f"{', '.join(HOLDOUT_BEARINGS)}；validation {len(y_va)} 行")
    print(f"参照：官方起点 681.16 ｜ 常数(训练集中位数) 470.9 ｜ 每轴承均值(作弊) 415.5")
    print(f"线参照（同一套数据、同一权重）：岭回归·原始特征 {ridge_raw['rmse']:.1f} ／ "
          f"岭回归·秩特征 {ridge_rank['rmse']:.1f}\n")

    # ---------- 候选配置 ----------
    # (名称, 初始化, 权重衰减, 是否对照)
    # 对照（原始特征·不加权）**不参与选型**，只用来量化"秩变换+等权"到底值多少分。
    if args.quick:
        cands = [("秩·等权·随机初始化·wd=1e-2", None, 1e-2, False)]
    else:
        cands = [
            ("秩·等权·随机初始化·wd=0", None, 0.0, False),
            ("秩·等权·随机初始化·wd=1e-3", None, 1e-3, False),
            ("秩·等权·随机初始化·wd=1e-2", None, 1e-2, False),
            ("秩·等权·热启动·wd=0", args.init, 0.0, False),
            ("秩·等权·热启动·wd=1e-3", args.init, 1e-3, False),
            ("原始·不加权·热启动·wd=0（对照）", args.init, 0.0, True),
        ]

    searched, finals = [], []
    for name, init_from, wd, is_control in cands:
        use_rank = "原始" not in name
        RANK_TABLE, RANKED = table if use_rank else None, use_rank
        xa = x_rank_tr if use_rank else train["x"]
        xb = x_rank_va if use_rank else valid["x"]
        weights = None if "不加权" in name else w

        info = select_epochs(name, xa, y_tr, b_tr, weights, paths, init_from, wd,
                             args.max_epochs)
        RANK_TABLE, RANKED = table if use_rank else None, use_rank
        fin = finalize(name, xa, y_tr, b_tr, xb, y_va, b_va, weights, paths,
                       init_from, wd, info["best_epoch"])
        searched.append({**info, "wd": wd, "rank": use_rank,
                         "weighted": weights is not None,
                         "control": is_control,
                         "init": "checkpoint" if init_from else "random"})
        finals.append(fin)
        print(f"[{name}]")
        print(f"  内部留出最优轮次 = 第 {info['best_epoch']} 轮"
              f"（holdout RMSE {info['holdout_rmse']:.1f}）")
        head = " ".join(f"{c['epoch']}:{c['holdout_rmse']:.0f}" for c in info["curve"][:10])
        print(f"  留出曲线: {head}{' ...' if len(info['curve']) > 10 else ''}")
        print(f"  → validation: RMSE {fin['rmse']:.2f}  MAE {fin['mae']:.2f}  "
              f"Bias {fin['bias']:+.1f}  秩相关 {fin['spearman']:+.3f}\n")

    # ---------- 选择：只按内部留出指标，对照不参与 ----------
    eligible = [i for i, s in enumerate(searched) if not s.get("control")]
    pick = min(eligible, key=lambda i: (searched[i]["holdout_rmse"], finals[i]["rmse"]))
    best, best_info = finals[pick], searched[pick]
    controls = [i for i, s in enumerate(searched) if s.get("control")]
    print("=" * 80)
    print("选型口径：**只按 train 内部留出轴承的 RMSE**（不看官方 validation）；对照组不参与选型")
    print(f"  选定：{best['name']}  轮次 {best['epochs']}  "
          f"内部留出 {best_info['holdout_rmse']:.1f}")
    print(f"  官方 validation：RMSE {best['rmse']:.2f}  MAE {best['mae']:.2f}  "
          f"（相对官方起点 681.16 变 {(best['rmse'] / 681.16 - 1) * 100:+.1f}%）")
    if controls:
        c = finals[controls[0]]
        print(f"  对照（原始特征·不加权）validation RMSE {c['rmse']:.2f} "
              f"→ 秩+等权带来 {(c['rmse'] - best['rmse']) / c['rmse'] * 100:+.1f}%")
    spread = [f for i, f in enumerate(finals) if not searched[i].get("control")]
    if len(spread) > 1:
        r = [f["rmse"] for f in spread]
        print(f"  敏感性：{len(spread)} 个候选的 validation RMSE 落在 "
              f"{min(r):.1f}–{max(r):.1f}（极差 {max(r) - min(r):.1f}）"
              f"→ {'结论对配置不敏感' if max(r) - min(r) < 20 else '配置有影响，需谨慎'}")

    # ---------- 保存 ----------
    out = T.MODEL_ROOT / "B" / "checkpoint_rank.npz"
    if out.exists():                     # Model.save 用 "xb" 模式，必须保证目标不存在
        out.unlink()
    best["model"].meta["preprocess_note"] = (
        "rank(ECDF fitted on official train.jsonl) + per-bearing equal weight + weight_decay; "
        f"recipe={best['name']}; epochs={best['epochs']}")
    # 标上 profile / target_unit，才能在 harness 里被官方 train_run 续训
    # （train_run 会校验：profile 一致、B 的 RUL 必须 target_unit=minutes）
    best["model"].meta["profile"] = "B"
    best["model"].meta["target_unit"] = "minutes"
    best["model"].save(out)

    report = {
        "task": "B",
        "baselines": {"official_checkpoint_rmse": 681.16,
                      "constant_train_median_rmse": 470.9,
                      "per_bearing_mean_oracle_rmse": 415.5,
                      "ridge_raw_rmse": ridge_raw["rmse"],
                      "ridge_rank_rmse": ridge_rank["rmse"]},
        "data": {"train_rows": int(len(y_tr)), "val_rows": int(len(y_va)),
                 "train_bearings": int(len(set(b_tr.tolist()))),
                 "holdout_bearings_for_selection": list(HOLDOUT_BEARINGS)},
        "config": {"hidden": HIDDEN, "lr": LR, "batch": BATCH,
                   "max_epochs": args.max_epochs,
                   "rank_quantiles": int(table.shape[1])},
        "search": [{k: v for k, v in s.items() if k != "curve"} for s in searched],
        "validation": [{k: v for k, v in f.items() if k not in ("model", "pred")}
                       for f in finals],
        "selected_by_holdout": best["name"],
        "selected_detail": {"epochs": best["epochs"], "rmse": best["rmse"],
                            "mae": best["mae"], "bias": best["bias"],
                            "spearman": best["spearman"],
                            "per_bearing": best["per_bearing"]},
        "checkpoint": str(out),
        "checkpoint_sha256": "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    out_json = PROJECT / "runs" / "B_improve_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
                        encoding="utf-8")

    print(f"\n模型已保存：{out}")
    print(f"报告：{out_json}")
    print("\n逐轴承（定稿模型）：")
    for b, v in best["per_bearing"].items():
        print(f"  {b:12s} n={v['n']:5d}  RMSE {v['rmse']:7.1f}  MAE {v['mae']:7.1f}  "
              f"Bias {v['bias']:+8.1f}")
    return report


if __name__ == "__main__":
    main()
