"""镜像统一入口：按 HARNESS_MODE 选择"闭环"还是"管道推理"。

    HARNESS_MODE=loop    （默认）跑自进化闭环：监测→归因→优化→守门→回滚，产物落 /output
    HARNESS_MODE=serve   官方 predict.py 协议：stdin JSONL 请求 → stdout JSONL 预测
    HARNESS_MODE=selftest 秒级自检（不训练，只验证代码/模型/数据可达）

平台若用 `docker run <image>` 且不传参数，就走 loop；若用管道喂样本，设 HARNESS_MODE=serve。
两种都给，是因为提交页没有明确说明数据投喂方式，宁可多做一手。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def selftest():
    import json
    import tasks as T
    print("[selftest] 检查 vendor / models / data ...")
    ok = True
    bases = {}
    for task in T.TASKS:
        vdir = T.VENDOR_ROOT / task
        ckpt = T.MODEL_ROOT / task / "checkpoint.npz"
        try:
            base = T.task_base(task)
        except FileNotFoundError:
            base = Path(T.DATA_ROOT or "/data")
        bases[task] = base
        flags = {"vendor": (vdir / "model.py").exists(), "model": ckpt.exists(),
                 "train.jsonl": (base / "train.jsonl").exists(),
                 "validation.jsonl": (base / "validation.jsonl").exists()}
        print(f"  {task}: " + "  ".join(f"{k}={'OK' if v else '缺失'}" for k, v in flags.items()))
        ok = ok and all(flags.values())

    import metrics, policy, runlog, serve                 # noqa: F401
    print(f"[selftest] import OK；DATA_ROOT={T.DATA_ROOT}；VENDOR={T.VENDOR_ROOT}")

    if ok:
        print("[selftest] 真实样本推理（每包 1 条，走官方 predict 协议）...")
        server = serve.Server()
        for task, base in bases.items():
            with (base / "validation.jsonl").open("r", encoding="utf-8-sig") as stream:
                row = json.loads(stream.readline())
            result = server.handle({"inputs": row["inputs"]})
            truth = row["answer"]
            got = (result["outputs"]["value"] if isinstance(result["outputs"], dict)
                   else max(result["outputs"], key=lambda d: d["score"])["label"])
            print(f"  {task}: truth={truth!r}  pred={got!r}")
    print("[selftest] 结果：" + ("通过" if ok else "有缺失"))
    return 0 if ok else 1


def main():
    mode = os.environ.get("HARNESS_MODE", "loop").strip().lower()
    if mode == "serve":
        import serve
        return serve.main(sys.argv[1:])
    if mode == "selftest":
        return selftest()
    import runner
    return runner.main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
