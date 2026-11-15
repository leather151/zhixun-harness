"""对"交付位"上的三个模型做一次干净的本地测量，用于回答"客观到底怎么样"。

规则：
- 只用官方 `validation.jsonl`，只测不改；
- 交付位（`models/<TASK>/checkpoint.npz`）与官方起点（`checkpoint_official_baseline.npz`）
  用完全相同的代码路径评测，方便对比；
- 指标口径与平台一致：A 用"仅观测到的类别"计入的 Macro-F1 + Accuracy；
  B/C 用 RMSE / MAE / bias / Pearson；C 另报 NASA 非对称评分。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP / "harness"))

import metrics as M          # noqa: E402
import tasks as T            # noqa: E402


def nasa(pred, y):
    d = pred - y
    return float(np.mean(np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)))


def _vendor_model(task):
    """按任务加载 vendor/<TASK>/model.py，用唯一模块名（避免 tasks.load_vendor 的单任务守卫）。"""
    import importlib.util
    name = f"_eval_vendor_{task}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, T.VENDOR_ROOT / task / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def evaluate(task, ckpt):
    rows = T.load_jsonl(T.task_base(task) / "validation.jsonl", T.TASKS[task]["classes"] or None)
    mod = _vendor_model(task)
    m = mod.Model.load(ckpt)
    # 交付位一律按"原始特征"喂：平台就是这么推理的
    raw = np.asarray(m.predict(rows["x"]))
    out = {"checkpoint": ckpt.name, "hidden": m.meta["hidden"],
           "params": m.meta["parameter_count"], "n": int(len(rows["y"])),
           "preprocess": (m.meta.get("preprocess") or {}).get("kind", "none")}
    if T.TASKS[task]["kind"] == "classification":
        pred = raw.argmax(1)
        out["macro_f1"] = M.macro_f1(rows["y"].astype(int), pred, len(T.TASKS[task]["classes"]))
        out["accuracy"] = float(np.mean(pred == rows["y"]))
        per = {}
        for c, name in enumerate(T.TASKS[task]["cn"]):
            sel = rows["y"].astype(int) == c
            if sel.sum():
                per[name] = {"n": int(sel.sum()), "recall": round(float(np.mean(pred[sel] == c)), 4)}
        out["per_class"] = per
    else:
        p = raw[:, 0] if raw.ndim > 1 else raw
        out.update({"rmse": M.rmse(p, rows["y"]), "mae": M.mae(p, rows["y"]),
                    "bias": M.bias(p, rows["y"]),
                    "pearson": float(np.corrcoef(p, rows["y"])[0, 1])})
        if task == "C":
            out["nasa"] = nasa(p, rows["y"])
        out["pred_range"] = [float(p.min()), float(p.max())]
        out["truth_range"] = [float(rows["y"].min()), float(rows["y"].max())]
    return out


def main():
    result = {}
    for task in T.TASKS:
        root = T.MODEL_ROOT / task
        pair = {}
        for label, name in (("delivered", "checkpoint.npz"),
                            ("baseline", "checkpoint_official_baseline.npz")):
            path = root / name
            if path.is_file():
                pair[label] = evaluate(task, path)
        if (root / "checkpoint_rank.npz").is_file() and task == "B":
            pair["rank_archive"] = None      # 秩模型需要外部变换，不能按原始特征评
        result[task] = pair

    print("=" * 78)
    for task, pair in result.items():
        d = pair.get("delivered")
        b = pair.get("baseline")
        print(f"\n【{task}】交付位 {d['checkpoint']}  hidden={d['hidden']}  参数={d['params']}  "
              f"预处理={d['preprocess']}  n={d['n']}")
        if T.TASKS[task]["kind"] == "classification":
            print(f"  交付版  Macro-F1={d['macro_f1']:.4f}  Acc={d['accuracy']:.4f}")
            if b:
                print(f"  基线版  Macro-F1={b['macro_f1']:.4f}  Acc={b['accuracy']:.4f}"
                      f"   → Δ Macro-F1 {d['macro_f1']-b['macro_f1']:+.4f}")
            print("  逐类召回（交付版）：",
                  {k: v["recall"] for k, v in d["per_class"].items()})
        else:
            unit = "分钟" if task == "B" else "循环"
            print(f"  交付版  RMSE={d['rmse']:.2f} MAE={d['mae']:.2f} bias={d['bias']:+.2f} "
                  f"ρ={d['pearson']:+.3f}" + (f" NASA={d['nasa']:.3e}" if "nasa" in d else ""))
            if b:
                print(f"  基线版  RMSE={b['rmse']:.2f} MAE={b['mae']:.2f} bias={b['bias']:+.2f} "
                      f"ρ={b['pearson']:+.3f}" + (f" NASA={b['nasa']:.3e}" if "nasa" in b else ""))
                print(f"  → Δ RMSE {d['rmse']-b['rmse']:+.2f} ({100*(d['rmse']-b['rmse'])/b['rmse']:+.1f}%)"
                      f"  Δ MAE {d['mae']-b['mae']:+.2f} ({100*(d['mae']-b['mae'])/b['mae']:+.1f}%)")
            print(f"  预测范围 {d['pred_range'][0]:.0f}–{d['pred_range'][1]:.0f}{unit} / "
                  f"真值范围 {d['truth_range'][0]:.0f}–{d['truth_range'][1]:.0f}{unit}")

    out = APP / "runs" / "delivered_eval.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[json] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
