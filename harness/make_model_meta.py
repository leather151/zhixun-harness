"""生成 models/<TASK>/meta.json 侧车文件：纯 JSON、不含权重，供平台/人工快速核对。

为什么需要：预检脚本读 .npz 需要 numpy；而"这个任务是什么、输入几维、有没有预处理、
参数多少、验证集指标多少"这类事实应当能用一个 json.load 直接拿到。
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(r"D:\pycharm\competition\models")

SPEC = {
    "A": {
        "default_checkpoint": "checkpoint_best.npz",
        "kind": "classification",
        "feature_count": 128,
        "input_path_hint": "inputs.features[0..127]",
        "classes": ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"],
        "units": "",
        "validation": {"macro_f1": 0.7268, "note": "官方 validation.jsonl，仅观测到的类别计入 Macro-F1（与官方口径一致）"},
        "recipe": "原始特征 · 类别权重 sqrt · wd=0 · epochs=2",
    },
    "B": {
        "default_checkpoint": "checkpoint_rank.npz",
        "kind": "regression",
        "feature_count": 34,
        "input_path_hint": "inputs.horizontal[0..16], inputs.vertical[0..16]",
        "classes": [],
        "units": "minutes",
        "validation": {"rmse": 369.47, "mae": 301.34, "bias": -110.3, "pearson_r": 0.655,
                       "note": "官方 validation.jsonl；对照常值基线 RMSE 470.9"},
        "recipe": "秩变换(ECDF,201分位,仅train拟合) + 按轴承等权 + 随机初始化 + wd=1e-2 + epochs=2",
    },
    "C": {
        "default_checkpoint": "checkpoint_best.npz",
        "kind": "regression",
        "feature_count": 24,
        "input_path_hint": "inputs.settings[0..2], inputs.sensors[0..20]",
        "classes": [],
        "units": "cycles",
        "validation": {"rmse": 51.56, "mae": 37.98, "nasa_score": 3.20e9,
                       "note": "官方 validation.jsonl；NASA 非对称评分，越小越好"},
        "recipe": "同初始化多轮打乱权重平均 · epochs=16 · 无预处理",
    },
}


def main():
    for task, spec in SPEC.items():
        ckpt = ROOT / task / spec["default_checkpoint"]
        assert ckpt.is_file(), ckpt
        import numpy as np
        z = np.load(ckpt, allow_pickle=False)
        md = json.loads(str(z["metadata"].item()))
        preprocess = md.get("preprocess") or {}
        out = {
            "task": task,
            "kind": spec["kind"],
            "profile": md.get("profile", task),
            "format": md.get("format"),
            "feature_count": spec["feature_count"],
            "input_path_hint": spec["input_path_hint"],
            "classes": spec["classes"],
            "target_unit": spec["units"],
            "hidden": md.get("hidden"),
            "parameter_count": md.get("parameter_count"),
            "default_checkpoint": spec["default_checkpoint"],
            "checkpoint_sha256": "sha256:" + hashlib.sha256(ckpt.read_bytes()).hexdigest(),
            "checkpoint_bytes": ckpt.stat().st_size,
            "preprocess": ("none" if not preprocess else
                           {"kind": preprocess.get("kind"),
                            "n_quantiles": preprocess.get("n_quantiles"),
                            "fitted_on": "official train.jsonl only",
                            "stored_in": "checkpoint metadata.preprocess.quantiles"}),
            "target_scaling": {"mean": md.get("target_mean"), "scale": md.get("target_scale")},
            "seed": md.get("seed"),
            "validation": spec["validation"],
            "recipe": spec["recipe"],
            "compliance": ["validation 只测不改", "预处理仅在 train 上拟合",
                           "id/meta 不进模型输入", "不自造标签"],
        }
        z.close()
        path = ROOT / task / "meta.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("wrote", path, path.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
