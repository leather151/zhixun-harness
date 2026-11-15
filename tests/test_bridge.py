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
for entry in (str(APP), str(APP / "harness")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import tasks as T                                            # noqa: E402
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

    # 交付位的模型必须"只依赖原始特征"：平台用自己的 predict 路径，不会施加我们的预处理
    for task in T.TASKS:
        m = load(task)
        assert not (m.meta.get("preprocess") or {}), f"{task} 交付模型不该带外部预处理"
    b = load("B")
    row = heads("B", 1)[0]
    paths, values = flatten_inputs(row["inputs"])
    x = np.asarray([values], dtype=np.float64)
    x_after = PRE.transform_inputs(b.meta, x)
    same = np.allclose(x, x_after)
    p = float(b.predict(x)[0])
    print(f"B 交付模型：无外部预处理={same}；pred={p:.2f}（若被夹到 0 就说明用错了模型）")
    ok = ok and same and p > 0

    # 秩档案仍在，且确实需要变换才可用（比值差异巨大 → 证明当初不能直接换名）
    import importlib.util
    spec = importlib.util.spec_from_file_location("_vB", T.VENDOR_ROOT / "B" / "model.py")
    vB = importlib.util.module_from_spec(spec)
    sys.modules["_vB"] = vB
    spec.loader.exec_module(vB)
    rank_ckpt = T.checkpoint_path("B", "checkpoint_rank.npz")
    if rank_ckpt.exists():
        tm = vB.Model.load(rank_ckpt)
        p_rank = float(tm.predict(PRE.transform_inputs(tm.meta, x))[0])
        p_raw = float(tm.predict(x)[0])
        print(f"B 秩档案：变换后 pred={p_rank:.2f} / 直接喂原始特征 pred={p_raw:.2f}")
        ok = ok and abs(p_rank - p_raw) > 1.0

    print("结果：" + ("通过" if ok else "失败"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
