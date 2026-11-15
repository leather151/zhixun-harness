"""A 包模型优化实验：类别加权 × 秩变换（配置搜索 + 定稿）

设计依据：`03_实验/数据分析/输出/A_分析.md`
  - 官方起点 Macro-F1 0.7140，但**验证集 109 条「乙醛」全部判错**（108 条判成「乙醇」），
    该类 F1 = 0；其余 5 类平均 0.8568 → 单类塌陷吃掉了 1/6 的 Macro-F1；
  - 乙醛在**同批训练集**上召回也只有 0.70 → 是"小类样本不足 + 类间边界不清"，不是纯漂移；
  - 逐 epoch 曲线第 10 轮后走平，最优轮是第 12 轮（0.7313），随包的是最后一轮（0.7140）；
  - 秩变换（每列分位）在 EDA 里把线性探针 0.327→0.619、1-NN 0.635→0.700，
    但对 MLP 是否有利**未验证** → 本轮实测。

纪律：
  - 官方 validation 只报一次；选配置/选轮次只用 **train(b01) 内部按类别分层的时间切分**；
  - 类别权重只用 fit 集的类频计算，不碰 validation。

用法：
    python harness/train_a.py                # 完整搜索 + 定稿
    python harness/train_a.py --quick        # 只跑定稿配置
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

import metrics as M       # noqa: E402
import preprocess as P    # noqa: E402
import tasks as T         # noqa: E402

HOLDOUT_SHARE = 0.25      # 每类按时间留最后 25% 做选型
LR = 0.003
BATCH = 128               # 官方配置；A 只有 445 行，batch 再大会一步走完全部
HIDDEN = 32
CLASSES = T.TASKS["A"]["classes"]
CN = T.TASKS["A"]["cn"]

RANK_TABLE = None
RANKED = False
PATHS = None


# ------------------------------------------------------------------ 工具

def class_weights(y, n_classes, mode="sqrt"):
    """类别权重：sqrt 平衡更稳（乙醛只有 30 条，全平衡容易过拟合）。"""
    counts = np.bincount(y.astype(int), minlength=n_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    if mode == "sqrt":
        w = np.sqrt(len(y) / (n_classes * counts))
    else:
        w = len(y) / (n_classes * counts)
    return w


def per_class_report(y_true, y_pred):
    rows = []
    for c, name in enumerate(CLASSES):
        sel = y_true == c
        support = int(sel.sum())
        recall = float(np.mean(y_pred[sel] == c)) if support else float("nan")
        tp = int(np.sum(sel & (y_pred == c)))
        fp = int(np.sum(~sel & (y_pred == c)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * recall / (prec + recall) if prec + recall else 0.0
        rows.append({"class": name, "cn": CN[c], "support": support,
                     "predicted": tp + fp, "precision": prec, "recall": recall, "f1": f1})
    return rows


def confusion(y_true, y_pred, n=6):
    return M.confusion(y_true.astype(int), y_pred.astype(int), n)


# ------------------------------------------------------------------ 训练

def build_model(x_fit, y_fit, init_from):
    mods = T.load_vendor("A")
    model = mods["model"].Model("classification", PATHS, CLASSES, x_fit, y_fit,
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
    return model


def split_fit_holdout(y, meta):
    """按类别分层的**时间**切分：每类按 source_index 排序，最后 25% 做选型。"""
    idx = np.asarray([int(m.get("source_index", i)) for i, m in enumerate(meta)])
    fit_idx, hold_idx = [], []
    for c in range(len(CLASSES)):
        rows = np.where(y == c)[0]
        rows = rows[np.argsort(idx[rows], kind="mergesort")]
        cut = max(1, int(round(len(rows) * (1 - HOLDOUT_SHARE))))
        if len(rows) <= 2:                       # 类别太小就全放进 fit
            fit_idx.extend(rows.tolist())
            continue
        fit_idx.extend(rows[:cut].tolist())
        hold_idx.extend(rows[cut:].tolist())
    return np.asarray(sorted(fit_idx)), np.asarray(sorted(hold_idx))


def select_epochs(name, x_tr, y_tr, y_all, w_sample, init_from, wd, max_epochs):
    fit_idx, hold_idx = split_fit_holdout(y_tr, META)
    model = build_model(x_tr[fit_idx], y_tr[fit_idx], init_from)
    w_fit = w_sample[fit_idx] if w_sample is not None else None
    curve = []
    for epoch in range(1, max_epochs + 1):
        model.fit(x_tr[fit_idx], y_tr[fit_idx], 1, sample_weight=w_fit, weight_decay=wd)
        pred = model.predict(x_tr[hold_idx]).argmax(1)
        f1 = M.macro_f1(y_tr[hold_idx].astype(int), pred, len(CLASSES), observed_only=False)
        acc = float(np.mean(pred == y_tr[hold_idx]))
        curve.append({"epoch": epoch, "holdout_macro_f1": f1, "holdout_accuracy": acc})
    best = max(curve, key=lambda r: r["holdout_macro_f1"])
    return {"name": name, "curve": curve, "best_epoch": best["epoch"],
            "holdout_macro_f1": best["holdout_macro_f1"]}


def finalize(name, x_tr, y_tr, x_va, y_va, w_sample, init_from, wd, epochs):
    model = build_model(x_tr, y_tr, init_from)
    model.fit(x_tr, y_tr, epochs, sample_weight=w_sample, weight_decay=wd)
    if RANKED:
        model.meta["preprocess"] = P.table_to_meta(RANK_TABLE)
    proba = model.predict(x_va)
    pred = proba.argmax(1)
    return {"name": name, "epochs": epochs, "model": model,
            "macro_f1": M.macro_f1(y_va.astype(int), pred, len(CLASSES), observed_only=False),
            "accuracy": float(np.mean(pred == y_va)),
            "per_class": per_class_report(y_va.astype(int), pred),
            "confusion": confusion(y_va, pred).tolist(),
            "confidence_mean": float(proba.max(1).mean())}


META = None


def drift_keep_mask(x, meta, share=0.30, thr=1.0):
    """用 **b01 内部前后时间片** 估计每个特征的"时间漂移"，剔除漂移过大的特征。

    完全不用 validation（连它的输入都不用），因此不违反"验证集只用于衡量"。
    依据：EDA 发现 F42 的跨批均值漂移达 11.5σ，这类特征会把 tanh 单元直接推到饱和。
    """
    idx = np.asarray([int(m.get("source_index", i)) for i, m in enumerate(meta)])
    order = np.argsort(idx, kind="mergesort")
    cut = max(1, int(round(len(order) * (1 - share))))
    early, late = x[order[:cut]], x[order[cut:]]
    sd = x.std(0)
    sd[sd < 1e-12] = 1
    z = (late.mean(0) - early.mean(0)) / sd
    keep = np.abs(z) <= thr
    if keep.sum() < 8:                      # 别把特征删光
        keep = np.argsort(-np.abs(z))[8:] >= 0
        keep = np.ones(x.shape[1], dtype=bool)
        keep[np.argsort(-np.abs(z))[:x.shape[1] - 8]] = False
    return keep, np.abs(z), z


def main(argv=None):
    global RANK_TABLE, RANKED, META, PATHS
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--init", default=str(T.MODEL_ROOT / "A" / "checkpoint.npz"))
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
            print("⚠ swanlab 未安装：只能记本地（PyCharm 解释器请选 anaconda2）")
        swan = L.open_swan(project=args.swan_project,
                           experiment="A-gas-train-improve",
                           config={"task": "A", "recipe": "class_weight + rank",
                                   "hidden": HIDDEN, "lr": LR, "batch": BATCH,
                                   "max_epochs": args.max_epochs,
                                   "holdout_share": HOLDOUT_SHARE})
        if swan["run"]:
            print(f"SwanLab：{swan['project']} / {swan['experiment']}（{swan['mode']}）")
            if swan["url"]:
                print(f"记录链接：{swan['url']}")
        else:
            print(f"⚠ SwanLab 未启用（{swan['error']}）")

    base_dir = T.task_base("A")
    train = T.load_jsonl(base_dir / "train.jsonl")
    valid = T.load_jsonl(base_dir / "validation.jsonl")
    paths = train["paths"]
    META = train["meta"]
    y_tr = np.asarray([CLASSES.index(v) for v in train["labels"]])
    y_va = np.asarray([CLASSES.index(v) for v in valid["labels"]])

    table = P.fit_rank_table(train["x"])
    x_rank_tr, x_rank_va = P.apply_rank(table, train["x"]), P.apply_rank(table, valid["x"])

    fit_idx, hold_idx = split_fit_holdout(y_tr, META)
    cw_fit = class_weights(y_tr[fit_idx], len(CLASSES))
    cw_all = class_weights(y_tr, len(CLASSES))
    w_cw_all = cw_all[y_tr]
    w_fit_for_select = class_weights(y_tr[fit_idx], len(CLASSES))[y_tr]

    print("=" * 80)
    print("A 包模型优化：类别加权 × 秩变换")
    print("=" * 80)
    print(f"train {len(y_tr)} 行（{np.bincount(y_tr, minlength=6).tolist()}）；"
          f"validation {len(y_va)} 行（{np.bincount(y_va, minlength=6).tolist()}）")
    print("训练集各类样本数：" + "、".join(f"{CN[c]}{int((y_tr == c).sum())}" for c in range(6)))
    print(f"fit/holdout 切分（按类分层的时间尾）：fit {len(fit_idx)} / holdout {len(hold_idx)}")
    print("类别权重（sqrt 平衡，fit 集统计）："
          + "、".join(f"{CN[c]}×{cw_fit[c]:.2f}" for c in range(6)))
    print(f"\n基线：官方起点 Macro-F1 0.7140 / Acc 0.8786 ｜ 最优轮(第12轮) 0.7313\n")

    if args.quick:
        cands = [("原始·类权·wd=0", args.init, 0.0, "all")]
    else:
        cands = [
            ("原始·不加权·wd=0", args.init, 0.0, "all"),        # 对照：应接近官方 0.7140
            ("原始·类权·wd=0", args.init, 0.0, "all"),
            ("原始·类权·去漂移σ>1.0", None, 0.0, "filter:1.0"),
            ("原始·类权·去漂移σ>1.5", None, 0.0, "filter:1.5"),
            ("原始·类权·去漂移σ>2.0", None, 0.0, "filter:2.0"),
            ("原始·类权·去漂移σ>2.5", None, 0.0, "filter:2.5"),
        ]

    # 特征漂移过滤（只用 b01 内部前后时间片，不碰 validation）
    keep_cache = {}
    for _, _, _, feat in cands:
        if feat.startswith("filter"):
            thr = float(feat.split(":")[1])
            keep_cache[thr] = drift_keep_mask(train["x"], META, thr=thr)[0]
    base_keep, zabs, zshift = drift_keep_mask(train["x"], META, thr=1.0)
    print(f"\nb01 内部时间漂移（前后 30% 对比）：")
    for thr in sorted(keep_cache):
        k = keep_cache[thr]
        print(f"  剔除 |z|>{thr}: 保留 {int(k.sum())}/{len(k)} 个特征")
    dropped = np.where(~base_keep)[0]
    if len(dropped):
        print("  |z|>1 被剔除的：" + "、".join(f"F{j + 1}({zshift[j]:+.1f}σ)" for j in dropped[:10])
              + (" ..." if len(dropped) > 10 else ""))

    searched, finals = [], []
    for name, init_from, wd, feat in cands:
        use_rank = "秩" in name
        use_weight = "类权" in name and "不加权" not in name
        RANK_TABLE, RANKED = (table if use_rank else None), use_rank
        xa = x_rank_tr if use_rank else train["x"]
        xb = x_rank_va if use_rank else valid["x"]
        keep = None
        if feat.startswith("filter"):
            keep = keep_cache[float(feat.split(":")[1])]
            xa, xb = xa[:, keep], xb[:, keep]
            PATHS = [paths[j] for j in np.where(keep)[0]]
        else:
            PATHS = paths
        w_sel = w_fit_for_select if use_weight else None
        w_all = w_cw_all if use_weight else None

        info = select_epochs(name, xa, y_tr, y_tr, w_sel, init_from, wd, args.max_epochs)
        RANK_TABLE, RANKED = (table if use_rank else None), use_rank
        fin = finalize(name, xa, y_tr, xb, y_va, w_all, init_from, wd, info["best_epoch"])
        if keep is not None:
            fin["model"].meta["preprocess"] = {"kind": "select",
                                              "feature_index": np.where(keep)[0].tolist()}
        searched.append({**info, "rank": use_rank, "weighted": use_weight, "wd": wd,
                         "feat": feat,
                         "init": "checkpoint" if init_from else "random"})
        finals.append(fin)
        print(f"[{name}]")
        print(f"  选型（train 内部留出）：最优第 {info['best_epoch']} 轮，"
              f"Macro-F1 {info['holdout_macro_f1']:.4f}")
        head = " ".join(f"{c['epoch']}:{c['holdout_macro_f1']:.3f}" for c in info["curve"][:12])
        print(f"  留出曲线: {head}{' ...' if len(info['curve']) > 12 else ''}")
        print(f"  → validation: Macro-F1 {fin['macro_f1']:.4f}  Acc {fin['accuracy']:.4f}  "
              f"置信度 {fin['confidence_mean']:.3f}")
        jiaquan = fin["per_class"][3]
        print(f"     乙醛：support {jiaquan['support']}  recall {jiaquan['recall']:.3f}  "
              f"F1 {jiaquan['f1']:.3f}\n")
        if swan["run"] is not None:
            tag = (name.replace("原始·", "raw_").replace("秩·", "rank_")
                       .replace("不加权·", "now_").replace("类权·", "cw_")
                       .replace("wd=", "wd"))
            try:
                for point in info["curve"]:
                    swan["run"].log({"holdout_macro_f1/" + tag: point["holdout_macro_f1"],
                                     "holdout_accuracy/" + tag: point["holdout_accuracy"]},
                                    step=point["epoch"])
                swan["run"].log({"val_macro_f1/" + tag: fin["macro_f1"],
                                 "val_accuracy/" + tag: fin["accuracy"],
                                 "val_yiquan_recall/" + tag: jiaquan["recall"],
                                 "val_yiquan_f1/" + tag: jiaquan["f1"],
                                 "best_epoch/" + tag: info["best_epoch"]},
                                step=args.max_epochs + len(searched))
            except Exception:                                  # noqa: BLE001
                pass

    # 选型：先在留出 Macro-F1 上取最优；**同分（差 <0.005，属噪声）时按先验打破平局**：
    # 优先"用了类别加权"的方案（类别不均衡是已知缺陷），其次不用筛特征、轮次更少（更简单）。
    best_h = max(s["holdout_macro_f1"] for s in searched)
    tied = [i for i, s in enumerate(searched) if s["holdout_macro_f1"] >= best_h - 0.005]
    pick = min(tied, key=lambda i: (not searched[i]["weighted"],
                                    searched[i]["feat"] != "all",
                                    searched[i]["best_epoch"]))
    best, best_info = finals[pick], searched[pick]
    print("=" * 80)
    print("选型口径：只按 train 内部留出 Macro-F1（不看官方 validation）；"
          "同分按先验打破平局（优先类别加权）")
    print(f"  留出最优 {best_h:.4f}，并列候选 {len(tied)} 个 → 选定：{best['name']}  "
          f"轮次 {best['epochs']}")
    print(f"  官方 validation：Macro-F1 {best['macro_f1']:.4f}  Acc {best['accuracy']:.4f}  "
          f"（相对官方起点 0.7140 变 {(best['macro_f1'] - 0.7140) / 0.7140 * 100:+.1f}%）")
    print("\n逐类：")
    for r in best["per_class"]:
        print(f"  {r['cn']:4s}({r['class']:12s}) support {r['support']:4d}  "
              f"预测 {r['predicted']:4d}  P {r['precision']:.3f}  R {r['recall']:.3f}  "
              f"F1 {r['f1']:.3f}")

    out = T.MODEL_ROOT / "A" / "checkpoint_best.npz"
    if out.exists():
        out.unlink()
    best["model"].meta["profile"] = "A"
    best["model"].meta["preprocess_note"] = (
        f"class_weight(sqrt) + rank optional; recipe={best['name']}; epochs={best['epochs']}")
    best["model"].save(out)

    report = {
        "task": "A",
        "baselines": {"official_macro_f1": 0.7140, "official_accuracy": 0.8786,
                      "best_epoch_macro_f1": 0.7313},
        "data": {"train_rows": int(len(y_tr)), "val_rows": int(len(y_va)),
                 "class_counts_train": np.bincount(y_tr, minlength=6).tolist(),
                 "class_counts_val": np.bincount(y_va, minlength=6).tolist(),
                 "fit_rows": int(len(fit_idx)), "holdout_rows": int(len(hold_idx))},
        "class_weights": {CN[c]: float(cw_all[c]) for c in range(6)},
        "search": [{k: v for k, v in s.items() if k != "curve"} for s in searched],
        "validation": [{k: v for k, v in f.items() if k != "model"} for f in finals],
        "selected": best["name"],
        "selected_detail": {"epochs": best["epochs"], "macro_f1": best["macro_f1"],
                            "accuracy": best["accuracy"],
                            "per_class": best["per_class"]},
        "checkpoint": str(out),
        "checkpoint_sha256": "sha256:" + hashlib.sha256(out.read_bytes()).hexdigest(),
        "swan": {k: v for k, v in swan.items() if k != "run"},
    }
    out_json = PROJECT / "runs" / "A_improve_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
                        encoding="utf-8")
    print(f"\n模型已保存：{out}\n报告：{out_json}")

    if swan["run"] is not None:
        try:
            swan["run"].log({"final/val_macro_f1": best["macro_f1"],
                             "final/val_accuracy": best["accuracy"],
                             "final/epochs": best["epochs"],
                             "db/official_macro_f1": 0.7140})
            swan["run"].finish()
        except Exception:                                      # noqa: BLE001
            pass
        if swan["url"]:
            print(f"\nSwanLab 记录：{swan['url']}")
    return report


if __name__ == "__main__":
    main()
