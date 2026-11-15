"""量化"优化效果到底看不看得出来"：配对 bootstrap 置信区间。

为什么需要：三点差值（比如 A 的 Macro-F1 +0.0128）不写区间就没有意义。
配对 bootstrap（同一批样本上重采样）能回答两个问题：
  1. 这个提升的方向稳不稳（95% CI 是否跨 0）；
  2. 幅度够不够大（相对我们自己的噪声带 ±0.02 / 相对基线）。

口径与平台一致：A 用"仅观测到的类别"计入的 Macro-F1；B/C 用 RMSE。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP / "harness"))

import tasks as T          # noqa: E402

RNG = np.random.default_rng(20261115)
N_BOOT = 3000


def vendor(task):
    import importlib.util
    name = f"_ci_vendor_{task}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, T.VENDOR_ROOT / task / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def macro_f1(y, pred, n_classes):
    """仅统计出现过的类别（与官方口径一致）。"""
    scores = []
    for c in range(n_classes):
        tp = np.sum((pred == c) & (y == c))
        fp = np.sum((pred == c) & (y != c))
        fn = np.sum((pred != c) & (y == c))
        if tp + fp + fn == 0:
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return float(np.mean(scores)) if scores else 0.0


def bootstrap_delta(metric_fn, pred_a, pred_b, y, n=N_BOOT):
    """返回 (观测差值 a-b, 95% CI 低, 高, P(a 更优))。"""
    obs = metric_fn(pred_a, y) - metric_fn(pred_b, y)
    diffs = np.empty(n)
    m = len(y)
    for i in range(n):
        idx = RNG.integers(0, m, m)
        diffs[i] = metric_fn(pred_a[idx], y[idx]) - metric_fn(pred_b[idx], y[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # 对 F1 越大越好、对 RMSE 越小越好，这里统一按"正值=交付版更差"来表达
    return float(obs), float(lo), float(hi), float(np.mean(diffs > 0))


def main():
    out = {}
    for task in T.TASKS:
        spec = T.TASKS[task]
        rows = T.load_jsonl(T.task_base(task) / "validation.jsonl", spec["classes"] or None)
        mod = vendor(task)
        deliv = mod.Model.load(T.MODEL_ROOT / task / "checkpoint.npz")
        base = mod.Model.load(T.MODEL_ROOT / task / "checkpoint_official_baseline.npz")
        y = rows["y"]
        pd = np.asarray(deliv.predict(rows["x"]))
        pb = np.asarray(base.predict(rows["x"]))

        if spec["kind"] == "classification":
            yc = y.astype(int)
            dd, bb = pd.argmax(1), pb.argmax(1)
            f = lambda p, t: macro_f1(t, p, len(spec["classes"]))  # noqa: E731
            obs_d, obs_b = f(dd, yc), f(bb, yc)
            diffs = np.empty(N_BOOT)
            m = len(yc)
            for i in range(N_BOOT):
                idx = RNG.integers(0, m, m)
                diffs[i] = f(dd[idx], yc[idx]) - f(bb[idx], yc[idx])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            rec = {"metric": "macro_f1", "delivered": round(obs_d, 4), "baseline": round(obs_b, 4),
                   "delta": round(obs_d - obs_b, 4),
                   "ci95": [round(float(lo), 4), round(float(hi), 4)],
                   "p_delivered_worse": round(float(np.mean(diffs < 0)), 3),
                   "n": int(m)}
            per = {}
            for c, name in enumerate(spec["cn"]):
                sel = yc == c
                if sel.sum():
                    per[name] = {"n": int(sel.sum()),
                                 "recall_delivered": round(float(np.mean(dd[sel] == c)), 3),
                                 "recall_baseline": round(float(np.mean(bb[sel] == c)), 3)}
            rec["per_class_recall"] = per
        else:
            pdf, pbf = pd.ravel().astype(float), pb.ravel().astype(float)
            rmse = lambda p, t: float(np.sqrt(np.mean((p - t) ** 2)))  # noqa: E731
            obs_d, obs_b = rmse(pdf, y), rmse(pbf, y)
            diffs = np.empty(N_BOOT)
            m = len(y)
            for i in range(N_BOOT):
                idx = RNG.integers(0, m, m)
                diffs[i] = rmse(pdf[idx], y[idx]) - rmse(pbf[idx], y[idx])
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            rec = {"metric": "rmse", "delivered": round(obs_d, 2), "baseline": round(obs_b, 2),
                   "delta": round(obs_d - obs_b, 2),
                   "delta_pct": round(100 * (obs_d - obs_b) / obs_b, 1),
                   "ci95": [round(float(lo), 2), round(float(hi), 2)],
                   "p_delivered_worse": round(float(np.mean(diffs > 0)), 3),
                   "n": int(m)}
        out[task] = rec

        print(f"\n【{task}】n={rec['n']}  bootstrap {N_BOOT} 次")
        if rec["metric"] == "macro_f1":
            print(f"  交付 {rec['delivered']:.4f}  基线 {rec['baseline']:.4f}  "
                  f"Δ={rec['delta']:+.4f}")
            print(f"  Δ 的 95% CI = [{rec['ci95'][0]:+.4f}, {rec['ci95'][1]:+.4f}]  "
                  f"P(交付更差)={rec['p_delivered_worse']:.3f}")
            for k, v in rec["per_class_recall"].items():
                print(f"    {k}: n={v['n']:4d}  交付 {v['recall_delivered']:.3f} / "
                      f"基线 {v['recall_baseline']:.3f}")
        else:
            print(f"  交付 {rec['delivered']:.2f}  基线 {rec['baseline']:.2f}  "
                  f"Δ={rec['delta']:+.2f}（{rec['delta_pct']:+.1f}%）")
            print(f"  Δ 的 95% CI = [{rec['ci95'][0]:+.2f}, {rec['ci95'][1]:+.2f}]  "
                  f"P(交付更差)={rec['p_delivered_worse']:.3f}")

    (APP / "runs" / "effect_ci.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[json] runs/effect_ci.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
