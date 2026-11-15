"""给 B 交付模型做"方差收缩"：把预测往训练均值拉，系数取 train 内部留出的相关系数。

为什么需要：交付位的 B 模型（原始特征蒸馏版）在官方 validation 上 ρ 只有 **0.165**，
意味着它带着 58% 的系统性低估（bias −271.7）。ρ 这么低的时候，最小化 RMSE 的最优线性
组合必然大幅偏向均值——硬用原始预测反而比"预测均值"还差。

做法：把输出围绕训练均值的偏差按 α 收缩：
    pred' = target_mean + α · (pred_raw − target_mean)
在 checkpoint 里就是 `target_scale *= α`（`Model.predict` 的形式为
`max(0, logits*target_scale + target_mean)`），不需要改任何代码。

**α 的来源（合规关键）**：取 **train 内部按轴承留出**（Bearing2_3 / Bearing3_3）上测得的
相关系数 ρ_holdout = 0.59 —— 这是"换了没见过的轴承后模型还剩多少可用信息"的训练侧估计，
**不看官方 validation**。（顺带说明：validation 上更优的 α 约 0.30，但那是拿 validation
调参，本轮不做；差异记在报告里。）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP / "harness"))

import tasks as T          # noqa: E402

ALPHA = 0.59               # 来自 train 内部留出 ρ
SRC = APP / "models" / "B" / "checkpoint.npz"


def main():
    mod = T.load_vendor("B")["model"]
    valid = T.load_jsonl(T.task_base("B") / "validation.jsonl")
    baseline = mod.Model.load(T.MODEL_ROOT / "B" / "checkpoint_official_baseline.npz")

    model = mod.Model.load(SRC)
    before = model.predict(valid["x"])
    mu = model.meta["target_mean"]
    rows = T.load_jsonl(T.task_base("B") / "validation.jsonl")
    del rows

    # 收缩：只动 target_scale
    old_scale = float(model.meta["target_scale"])
    model.meta["target_scale"] = old_scale * ALPHA
    model.meta["shrinkage"] = {
        "alpha": ALPHA,
        "rule": "alpha = Pearson ρ on train-internal leave-bearing-out holdout (Bearing2_3/3_3)",
        "why": "raw 特征交付版 ρ 低，硬用原始预测不如向均值收缩",
        "note": "validation 上更优的 α≈0.30 未采用（那等于拿 validation 调参）",
    }
    model.meta["preprocess_note"] = (
        f"raw 特征 · 蒸馏自秩模型 · hidden={model.meta['hidden']} · "
        f"方差收缩 α={ALPHA}（train 内部留出 ρ）")

    def met(p):
        p = np.asarray(p, dtype=float).ravel()
        e = p - valid["y"]
        return {"rmse": round(float(np.sqrt((e ** 2).mean())), 2),
                "mae": round(float(np.abs(e).mean()), 2),
                "bias": round(float(e.mean()), 2),
                "pearson": round(float(np.corrcoef(p, valid["y"])[0, 1]), 3)}

    after = met(model.predict(valid["x"]))
    print("收缩前:", met(before))
    print("收缩后:", after)
    print("基线    :", met(baseline.predict(valid["x"])))

    SRC.unlink()
    model.save(SRC)
    print(f"[save] {SRC} ({SRC.stat().st_size} bytes)")

    report = {"alpha": ALPHA, "before": met(before), "after": after,
              "baseline": met(baseline.predict(valid["x"])),
              "target_mean": mu, "old_scale": old_scale,
              "new_scale": model.meta["target_scale"]}
    (APP / "runs" / "B_shrinkage_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[save] runs/B_shrinkage_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
