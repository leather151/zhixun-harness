"""C 包模型优化实验：早停 × 同初值多 shuffle 集成（权重平均，产出单个检查点）

设计依据：`03_实验/数据分析/输出/C_分析.md`
  - 官方起点 RMSE 52.87 / MAE 38.90 / NASA 5.82e9；常数基线 79.59；
  - 逐发动机秩相关均值 0.856（趋势学到了），但**逐发动机 RMSE 中位数只要 40.74**
    → 整体指标被少数几台拉高；
  - FD004 用 37.8% 的样本吃掉 **54%** 的平方误差；RUL>200 的样本占 **96.3%** 的 NASA 罚分；
  - **25-NN 局部平滑就有 50.95**（比基线好）→ 说明基线模型"太尖/欠平滑"，集成是对症的；
  - 已证否（不再试）：秩变换 55.83、高 RUL 降权 54.03、log1p 60.70。

本轮只做两件事：
  1. **早停**：轮次由 train 内部留出**发动机**选出（不使用官方 validation 选型）；
  2. **集成**：K 个模型**共享同一初值**（都从官方检查点拷贝权重），只用不同的 shuffle 种子训练，
     最后对**权重做平均** → 仍然是一个 `Model`，可以直接被 harness / serve 加载，不需要复合结构。

用法：
    python harness/train_c.py --swan on
    python harness/train_c.py --quick
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

import metrics as M      # noqa: E402
import tasks as T        # noqa: E402

HOLDOUT_EVERY = 5          # 每 5 台发动机留 1 台做选型（约 20%）
MAX_EPOCHS = 30
LR = 0.002                 # 官方配置
BATCH = 256                # 官方配置
HIDDEN = 32


def rmse(p, y):
    return float(np.sqrt(np.mean((np.asarray(p) - np.asarray(y)) ** 2)))


def mae(p, y):
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(y))))


def engine_key(meta):
    return np.array([f'{m["source_subset"]}_u{m["unit"]:03d}' for m in meta], dtype=object)


def split_engines(units):
    """按发动机留出（每 5 台留 1 台），完全不跨发动机泄漏。"""
    uniq = sorted(set(units.tolist()))
    hold = set(uniq[::HOLDOUT_EVERY])
    mask = np.array([u in hold for u in units])
    return ~mask, mask, sorted(hold)


def build(hidden, seed, x, y, paths, init_from=None):
    mods = T.load_vendor("C")
    model = mods["model"].Model("regression", paths, [], x, y,
                               hidden=hidden, seed=seed, learning_rate=LR, batch_size=BATCH)
    if init_from is not None:
        base = mods["model"].Model.load(init_from)
        if base.params["w1"].shape != model.params["w1"].shape:
            raise ValueError("权重形状不一致，无法热启动")
        for key in model.params:
            model.params[key] = base.params[key].copy()
    return model


def select_epochs(x_fit, y_fit, x_hold, y_hold, paths, hidden, seeds, init_from, max_epochs):
    """在内部留出**发动机**上逐 epoch 评估；集成时取各成员的均值曲线。"""
    models = [build(hidden, s, x_fit, y_fit, paths, init_from) for s in seeds]
    curve = []
    for epoch in range(1, max_epochs + 1):
        for m in models:
            m.fit(x_fit, y_fit, 1)
        preds = [m.predict(x_hold) for m in models]
        mean_pred = np.mean(preds, axis=0)
        curve.append({"epoch": epoch,
                      "holdout_rmse": rmse(mean_pred, y_hold),
                      "member_rmse": [rmse(p, y_hold) for p in preds]})
    best = min(curve, key=lambda r: r["holdout_rmse"])
    return curve, best


def fit_final(x_tr, y_tr, paths, hidden, seeds, init_from, epochs):
    models = [build(hidden, s, x_tr, y_tr, paths, init_from) for s in seeds]
    for m in models:
        m.fit(x_tr, y_tr, epochs)
    # 权重平均：因为共享同一初值、只差 shuffle 顺序，各成员落在同一盆地，平均是安全的
    avg = build(hidden, seeds[0], x_tr, y_tr, paths, init_from)
    for key in avg.params:
        avg.params[key] = np.mean([m.params[key] for m in models], axis=0)
    avg.meta["ensemble"] = {"n": len(models), "seeds": list(seeds), "epochs": epochs,
                            "method": "same-init multi-shuffle weight averaging"}
    return avg, models


def evaluate(model, x_va, y_va, subsets):
    pred = model.predict(x_va)
    out = {"rmse": rmse(pred, y_va), "mae": mae(pred, y_va),
           "bias": float(np.mean(pred - y_va)), "nasa": M.nasa_score(pred, y_va),
           "spearman": M.spearman(pred, y_va)}
    per = {}
    for s in sorted(set(subsets.tolist())):
        sel = subsets == s
        per[s] = {"n": int(sel.sum()), "rmse": rmse(pred[sel], y_va[sel]),
                  "mae": mae(pred[sel], y_va[sel]), "bias": float(np.mean(pred[sel] - y_va[sel]))}
    out["per_subset"] = per
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--init", default=str(T.MODEL_ROOT / "C" / "checkpoint.npz"))
    ap.add_argument("--swan", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--swan-project", default=None)
    args = ap.parse_args(argv)

    import platform
    print(f"解释器：{sys.executable}")
    print(f"Python {platform.python_version()}  numpy {np.__version__}")
    swan = {"run": None, "url": None, "error": None, "mode": None, "project": None,
            "experiment": None}
    if args.swan != "off":
        import runlog as L
        try:
            import swanlab as _sw
            print(f"swanlab {_sw.__version__}")
        except Exception:                                       # noqa: BLE001
            print("⚠ swanlab 未安装：只能记本地")
        swan = L.open_swan(project=args.swan_project,
                           experiment="C-cmapss-train-improve",
                           config={"task": "C", "recipe": "early_stop + same-init ensemble",
                                   "hidden": HIDDEN, "lr": LR, "batch": BATCH,
                                   "holdout_every": HOLDOUT_EVERY})
        if swan["run"]:
            print(f"SwanLab：{swan['project']} / {swan['experiment']}（{swan['mode']}）")
            if swan["url"]:
                print(f"记录链接：{swan['url']}")
        else:
            print(f"⚠ SwanLab 未启用（{swan['error']}）")

    base_dir = T.task_base("C")
    train = T.load_jsonl(base_dir / "train.jsonl")
    valid = T.load_jsonl(base_dir / "validation.jsonl")
    paths = train["paths"]
    y_tr, y_va = train["y"], valid["y"]
    u_tr = engine_key(train["meta"])
    sub_va = np.array([m["source_subset"] for m in valid["meta"]], dtype=object)
    fit_mask, hold_mask, hold_units = split_engines(u_tr)
    x_fit, y_fit = train["x"][fit_mask], y_tr[fit_mask]
    x_hold, y_hold = train["x"][hold_mask], y_tr[hold_mask]

    print("=" * 84)
    print("C 包模型优化：早停 × 同初值多 shuffle 集成（权重平均）")
    print("=" * 84)
    print(f"train {len(y_tr)} 行 / {len(set(u_tr.tolist()))} 台；validation {len(y_va)} 行")
    print(f"内部留出（每 {HOLDOUT_EVERY} 台留 1）：fit {int(fit_mask.sum())} 行 / "
          f"holdout {int(hold_mask.sum())} 行（{len(hold_units)} 台）")
    print("参照：官方 52.87 ｜ 常数 79.59 ｜ 25-NN 50.95 ｜ 历史最优轮（第14轮）51.85\n")

    if args.quick:
        # 定稿配置：单模型 + 早停（集成只多 0.19 分却要 5 倍训练，性价比不划算）
        cands = [("单模型·热启动·h32·早停", HIDDEN, [42], args.init)]
    else:
        cands = [
            ("单模型·热启动·h32（对照）", HIDDEN, [42], args.init),
            ("集成5·热启动·h32", HIDDEN, [42, 43, 44, 45, 46], args.init),
            ("集成5·随机初始化·h32", HIDDEN, [42, 43, 44, 45, 46], None),
            # 注意：h48 无法从 h32 的检查点热启动（权重形状不同），只能随机初始化；
            # 而"随机初始化 + 权重平均"是无效的（各成员落在不同盆地），所以只作参照。
            ("集成5·随机初始化·h48（参照）", 48, [42, 43, 44, 45, 46], None),
        ]

    searched, finals = [], []
    for name, hidden, seeds, init_from in cands:
        curve, best = select_epochs(x_fit, y_fit, x_hold, y_hold, paths, hidden, seeds,
                                    init_from, args.max_epochs)
        model, members = fit_final(train["x"], y_tr, paths, hidden, seeds, init_from,
                                   best["epoch"])
        fin = evaluate(model, valid["x"], y_va, sub_va)
        # 参考：如果用"预测平均"（会得到多模型，不便交付），能到多少
        pred_ens = np.mean([m.predict(valid["x"]) for m in members], axis=0)
        ens_rmse = rmse(pred_ens, y_va)
        searched.append({"name": name, "epochs": best["epoch"],
                         "holdout_rmse": best["holdout_rmse"], "hidden": hidden,
                         "seeds": list(seeds), "init": "checkpoint" if init_from else "random"})
        finals.append({**fin, "name": name, "epochs": best["epoch"],
                       "pred_ensemble_rmse": ens_rmse})
        print(f"[{name}]")
        print(f"  选型（内部留出）：最优第 {best['epoch']} 轮，holdout RMSE "
              f"{best['holdout_rmse']:.2f}")
        head = " ".join(f"{c['epoch']}:{c['holdout_rmse']:.1f}" for c in curve[:12])
        print(f"  留出曲线: {head}{' ...' if len(curve) > 12 else ''}")
        print(f"  → validation: RMSE {fin['rmse']:.2f}  MAE {fin['mae']:.2f}  "
              f"Bias {fin['bias']:+.2f}  NASA {fin['nasa']:.3e}")
        print(f"     权重平均 vs 预测平均：{fin['rmse']:.2f} vs {ens_rmse:.2f}")
        print("     逐子集：" + "  ".join(f"{s} {v['rmse']:.1f}"
                                        for s, v in fin["per_subset"].items()) + "\n")
        if swan["run"] is not None:
            tag = name.replace("·", "_").replace("（对照）", "")
            try:
                for c in curve:
                    swan["run"].log({f"holdout_rmse/{tag}": c["holdout_rmse"]},
                                    step=c["epoch"])
                swan["run"].log({f"val_rmse/{tag}": fin["rmse"],
                                 f"val_mae/{tag}": fin["mae"],
                                 f"val_nasa/{tag}": fin["nasa"],
                                 f"best_epoch/{tag}": best["epoch"],
                                 f"pred_ensemble_rmse/{tag}": ens_rmse},
                                step=args.max_epochs + len(searched))
            except Exception:                                   # noqa: BLE001
                pass

    best_i = min(range(len(searched)), key=lambda i: searched[i]["holdout_rmse"])
    best, best_info = finals[best_i], searched[best_i]
    print("=" * 84)
    print("选型口径：只按 train 内部留出**发动机**的 RMSE（不看官方 validation）")
    print(f"  选定：{best['name']}  轮次 {best['epochs']}")
    print(f"  官方 validation：RMSE {best['rmse']:.2f}  MAE {best['mae']:.2f}  "
          f"（相对官方起点 52.87 变 {(best['rmse'] / 52.87 - 1) * 100:+.1f}%）")
    print(f"  逐子集：" + "  ".join(f"{s} {v['rmse']:.1f}"
                                for s, v in best["per_subset"].items()))

    # 保存（权重平均后的单模型）
    out = T.MODEL_ROOT / "C" / "checkpoint_best.npz"
    if out.exists():
        out.unlink()
    model_out = fit_final(train["x"], y_tr, paths, best_info["hidden"],
                          best_info["seeds"], best_info["init"] == "checkpoint" and args.init
                          or None, best["epochs"])[0]
    model_out.meta["profile"] = "C"
    model_out.meta["preprocess_note"] = (f"same-init multi-shuffle weight averaging "
                                         f"(n={len(best_info['seeds'])}); "
                                         f"epochs={best['epochs']}")
    model_out.save(out)

    report = {
        "task": "C",
        "baselines": {"official_rmse": 52.87, "constant_rmse": 79.59,
                      "knn25_rmse": 50.95, "best_epoch_rmse": 51.85},
        "data": {"train_rows": int(len(y_tr)), "val_rows": int(len(y_va)),
                 "fit_rows": int(fit_mask.sum()), "holdout_rows": int(hold_mask.sum()),
                 "holdout_engines": len(hold_units)},
        "search": searched,
        "validation": finals,
        "selected": best["name"],
        "checkpoint": str(out),
        "checkpoint_sha256": "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest(),
        "swan": {k: v for k, v in swan.items() if k != "run"},
    }
    out_json = PROJECT / "runs" / "C_improve_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
                        encoding="utf-8")
    print(f"\n模型已保存：{out}\n报告：{out_json}")
    if swan["run"] is not None:
        try:
            swan["run"].log({"final/val_rmse": best["rmse"], "final/val_mae": best["mae"],
                             "final/val_nasa": best["nasa"],
                             "db/official_rmse": 52.87, "db/knn25_rmse": 50.95})
            swan["run"].finish()
        except Exception:                                       # noqa: BLE001
            pass
        if swan["url"]:
            print(f"\nSwanLab 记录：{swan['url']}")
    return report


if __name__ == "__main__":
    main()
