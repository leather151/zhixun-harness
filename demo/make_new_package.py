"""造一个"新数据包"（模拟评测换数据集）：7 维传感器字段 + 4 类，与 A/B/C 完全不同。

用来验证两件事：
1. `python -m baselines.train` 在未知 schema 上能不能自适应建网并训练；
2. `baselines.predict` 能不能按 schema 找到刚训出来的模型并预测。
"""
import json
import random
from pathlib import Path

OUT = Path(r"C:\Users\lenovo\AppData\Local\Temp\newpkg")
FIELDS = ("temp", "vib", "curr", "press", "flow", "rpm", "torque")
CLASSES = ("Normal", "Wear", "Crack", "Loose")


def row(i):
    random.seed(i)
    feats = {f: round(random.uniform(0, 10), 4) for f in FIELDS}
    # 标签由一个清晰但不平凡的组合规则决定（可学，但需要模型真的拟合）
    level = feats["temp"] * 0.6 + feats["vib"] * 0.4
    ans = CLASSES[min(3, int(level / 2.5))]
    return {"id": f"new-{i:04d}", "inputs": {"sensors": feats}, "answer": ans,
            "meta": {"source": "synthetic-package", "unit": "n/a"}}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", 400), ("validation", 160)):
        path = OUT / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for i in range(n):
                fh.write(json.dumps(row(i if split == "train" else i + 10000),
                                    ensure_ascii=False) + "\n")
        print(f"{path}  {path.stat().st_size} bytes")
    print(f"字段：{FIELDS}；类别：{CLASSES}；维度 7（A/B/C 分别是 128/34/24）")


if __name__ == "__main__":
    main()
