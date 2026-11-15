"""把 B 包的"秩变换 + 等权"成果**蒸馏**进一个只吃原始特征的模型。

为什么必须做这一步：
平台的评测路径是「加载 `models/<TASK>/checkpoint.npz` → `Model.predict(原始特征)`」，
**不会**施加我们 checkpoint metadata 里的秩变换（实测：平台报的 B 指标 681.159833 / 563.105841
正是官方基线 `checkpoint.npz` 的数字，而我们的秩模型是 369.47 / 301.34）。
如果直接把秩模型改名成 checkpoint.npz，平台按原始特征推理会得到 ≈0 的预测（ReLU 夹断），
RMSE 会比基线还差 —— 那会踩"不得低于静态基线"的红线。

所以：让秩模型当**教师**，输出训练集上的预测；学生只吃原始特征去拟合教师。
学生学的是 f_rank(x) 在数据分布上的近似，不需要在权重里复现分位映射。

合规：选型只用 **train 内部按轴承留出**（Bearing2_3 / Bearing3_3），官方 validation 只在
定稿后**测一次**报告。
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


def load_B():
    base = T.task_base("B")
    train = T.load_jsonl(base / "train.jsonl")
    valid = T.load_jsonl(base / "validation.jsonl")
    return train, valid


def bearing_of(train):
    return np.array([str(m.get("bearing", "?")) for m in train["meta"]], dtype=object)


def sample_weight_equal_per_bearing(groups):
    uniq, counts = np.unique(groups, return_counts=True)
    per = {g: 1.0 / c for g, c in zip(uniq, counts)}
    w = np.array([per[g] for g in groups], dtype=np.float64)
    return w / w.mean()


def metrics(pred, y):
    e = pred - y
    return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "bias": float(np.mean(e)), "pearson": float(np.corrcoef(pred, y)[0, 1])}


def main():
    mod = T.load_vendor("B")["model"]
    train, valid = load_B()
    groups = bearing_of(train)
    print(f"[data] train {len(train['y'])} 行 / {len(set(groups))} 个轴承；valid {len(valid['y'])} 行")

    # ---- 教师：现有秩模型 ----
    teacher = mod.Model.load(T.checkpoint_path("B", "checkpoint_rank.npz"))
    t_pred_train = teacher.predict(P.transform_inputs(teacher.meta, train["x"]))
    t_pred_valid = teacher.predict(P.transform_inputs(teacher.meta, valid["x"]))
    print("[teacher] 秩模型：train 预测已生成；validation(测量)",
          {k: round(v, 2) for k, v in metrics(t_pred_valid, valid["y"]).items()})

    # ---- 选型：train 内部按轴承留出 ----
    is_hold = np.array([g in HOLDOUT_GROUPS for g in groups])
    tr_idx = np.where(~is_hold)[0]
    ho_idx = np.where(is_hold)[0]
    print(f"[split] 拟合 {len(tr_idx)} 行 / 留出 {len(ho_idx)} 行（{sorted(set(groups[ho_idx]))}）")

    w_all = sample_weight_equal_per_bearing(groups)
    best = None
    print("\n[search] hidden × epochs（权重衰减 1e-2，按轴承等权，目标=教师预测）")
    for hidden in (64, 128):
        for epochs in (10, 30, 60):
            for wd in (1e-2, 1e-3):
                m = mod.Model("regression", train["paths"], [], train["x"][tr_idx],
                              t_pred_train[tr_idx], hidden=hidden, seed=42,
                              learning_rate=3e-3, batch_size=128)
                m.fit(train["x"][tr_idx], t_pred_train[tr_idx], epochs=epochs,
                      sample_weight=w_all[tr_idx], weight_decay=wd)
                ho = metrics(m.predict(train["x"][ho_idx]), train["y"][ho_idx])
                # 选型看"学生 vs 教师"在留出集上的差距 + 与真值的误差
                gap = metrics(m.predict(train["x"][ho_idx]), t_pred_train[ho_idx])
                print(f"  hidden={hidden:3d} epochs={epochs:2d} wd={wd:g}  "
                      f"留出 RMSE(对真值)={ho['rmse']:8.2f}  MAE={ho['mae']:8.2f}  "
                      f"bias={ho['bias']:+8.2f}  对教师 RMSE={gap['rmse']:7.2f}")
                row = {"hidden": hidden, "epochs": epochs, "wd": wd,
                       "holdout": ho, "gap": gap}
                if best is None or ho["rmse"] < best["holdout"]["rmse"]:
                    best = row
    print(f"\n[best] {best['hidden']}/{best['epochs']}/{best['wd']}  "
          f"留出 RMSE={best['holdout']['rmse']:.2f}")

    # ---- 定稿：全量 train 重训 → 官方 validation 测一次 ----
    final = mod.Model("regression", train["paths"], [], train["x"], t_pred_train,
                      hidden=best["hidden"], seed=42, learning_rate=3e-3, batch_size=128)
    final.fit(train["x"], t_pred_train, epochs=best["epochs"],
              sample_weight=w_all, weight_decay=best["wd"])
    final.meta["preprocess_note"] = (
        f"raw 特征 · 蒸馏自秩模型(教师 RMSE 369.47) · hidden={best['hidden']} "
        f"epochs={best['epochs']} wd={best['wd']:g} · 按轴承等权 · 无外部预处理")
    final.meta["distill_from"] = "models/B/checkpoint_rank.npz"
    final.meta["profile"] = "B"
    final.meta["target_unit"] = "minutes"

    val = metrics(final.predict(valid["x"]), valid["y"])
    print("\n[定稿] 官方 validation（只测一次）：",
          {k: round(v, 2) for k, v in val.items()})

    # 对照：基线与"直接对真值训练"的原始特征模型
    base = mod.Model.load(T.checkpoint_path("B", "checkpoint.npz"))
    base_val = metrics(base.predict(valid["x"]), valid["y"])
    print("[对照] 官方基线 checkpoint.npz：", {k: round(v, 2) for k, v in base_val.items()})

    out = APP / "models" / "B" / "checkpoint_raw_distilled.npz"
    if out.exists():
        out.unlink()
    final.save(out)
    print(f"\n[save] {out} ({out.stat().st_size} bytes)")

    report = {"teacher_validation": metrics(t_pred_valid, valid["y"]),
              "student_validation": val, "baseline_validation": base_val,
              "selected": best, "checkpoint": str(out.relative_to(APP))}
    (APP / "runs" / "B_distill_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[save] runs/B_distill_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
