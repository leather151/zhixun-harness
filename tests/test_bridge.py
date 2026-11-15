"""本机自测：一个进程内同时加载 A/B/C，验证 `baselines.predict` 桥的三件事。

1. 三条任务流共存不互相污染（官方 vendor 的 predict.py 这点做不到）；
2. 检查点自带的预处理真的生效（B 包的秩变换分位表）；
3. 桥的输出与官方 predict 协议同形（regression → {"outputs": {"value": ...}}）。

运行：
    python tests/test_bridge.py
"""
import json
import sys
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parent.parent
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import tasks as T                                            # noqa: E402  （harness/ 在 sys.path）
from baselines.predict import flatten_inputs, load, predict, predict_auto  # noqa: E402
import preprocess as PRE                                     # noqa: E402


def heads(task, n=3):
    path = T.task_base(task) / "validation.jsonl"
    with path.open("r", encoding="utf-8-sig") as fh:
        return [json.loads(next(fh)) for _ in range(n)]


def main():
    ok = True
    for task in T.TASKS:
        rows = heads(task)
        model = load(task)
        kind = model.meta["task"]
        print(f"[{task}] paths={len(model.meta['paths'])} hidden={model.meta['hidden']} "
              f"params={model.meta['parameter_count']} "
              f"preprocess={(model.meta.get('preprocess') or {}).get('kind')}")
        for row in rows:
            out = predict(model, row["inputs"])
            if kind == "regression":
                print(f"    answer={float(row['answer']):9.2f}  pred={out['outputs']['value']:9.2f}")
            else:
                top = max(out["outputs"], key=lambda o: o["score"])
                print(f"    answer={row['answer']:14s} pred={top['label']:14s} "
                      f"score={top['score']:.4f}")
        same = predict_auto(rows[0]["inputs"]) == predict(model, rows[0]["inputs"])
        print(f"    自动识别任务与显式指定一致：{'OK' if same else '不一致'}")
        ok = ok and same

    # B 包的秩变换必须真的生效
    b = load("B")
    row = heads("B", 1)[0]
    paths, values = flatten_inputs(row["inputs"])
    x = np.asarray([values], dtype=np.float64)
    x_rank = PRE.transform_inputs(b.meta, x)
    p_rank, p_raw = float(b.predict(x_rank)[0]), float(b.predict(x)[0])
    print(f"B 秩变换：{np.round(x[0][:3], 4)} -> {np.round(x_rank[0][:3], 4)}；"
          f"pred 秩={p_rank:.2f} / pred 原始={p_raw:.2f}")
    ok = ok and abs(p_rank - p_raw) > 1e-9

    print("结果：" + ("通过" if ok else "失败"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
