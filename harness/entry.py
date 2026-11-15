"""镜像统一入口：按 HARNESS_MODE 选择"闭环"还是"管道推理"。

    HARNESS_MODE=loop    （默认）跑自进化闭环：监测→归因→优化→守门→回滚，产物落 /model 与 /output
    HARNESS_MODE=serve   官方 predict.py 协议：stdin JSONL 请求 → stdout JSONL 预测
    HARNESS_MODE=selftest 秒级自检（不训练，只验证代码/模型/数据可达）

平台若用 `docker run <image>` 且不传参数，就走 loop；若用管道喂样本，设 HARNESS_MODE=serve。
两种都给，是因为提交页没有明确说明数据投喂方式，宁可多做一手。

**兜底交付**（重要）：loop 模式下入口第一步就先把镜像内自带的检查点落成交付产物，
再跑闭环；闭环失败（/data 未挂载、数据为空、超预算、父检查点 schema 不符……）不改变
"平台一定拿得到模型"这件事，容器仍以 0 退出并把失败原因写进 manifest.failure。
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _flag(argv, name):
    """从命令行里取 `--xxx value` / `--xxx=value`（入口只做兜底交付，不想重复整套 argparse）。"""
    for i, item in enumerate(argv):
        if item == name and i + 1 < len(argv):
            return argv[i + 1]
        if item.startswith(name + "="):
            return item.split("=", 1)[1]
    return None


def _tasks(argv):
    """`--task all` → None（三个都做）；`--task B` → ["B"]。"""
    task = _flag(argv, "--task")
    if task in (None, "all"):
        return None
    return [t.strip() for t in task.split(",") if t.strip()]


def selftest():
    import json
    import tasks as T
    print("[selftest] 检查 vendor / models / data ...")
    ok = True
    bases = {}
    for task in T.TASKS:
        vdir = T.VENDOR_ROOT / task
        ckpt = T.checkpoint_path(task)          # 用真正会交付的那个检查点（B 是秩变换版）
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

    import metrics, policy, runlog, serve, deliver            # noqa: F401
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


def loop(argv):
    """先兜底交付，再跑闭环；闭环出错也保证产物已在盘上。"""
    import deliver
    import runner

    tasks = _tasks(argv)
    out = _flag(argv, "--out")
    model_out = _flag(argv, "--model-out")
    deliver.ensure_bundled(tasks=tasks, out=out, model_out=model_out)
    try:
        result = runner.main(argv)
    except SystemExit as exc:                                 # argparse 之类
        code = exc.code
        return code if isinstance(code, int) else 0
    except BaseException as exc:                              # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        print(f"[loop] 闭环未完成：{reason}", file=sys.stderr)
        traceback.print_exc()
        deliver.ensure_bundled(tasks=tasks, out=out, model_out=model_out, failure=reason,
                              skip_if_evolved=True)
        print(f"[loop] 已按兜底交付处理：内置模型仍在 /model/<TASK>/checkpoint.npz")
        return 0
    # runner.main 返回的是 summary dict —— 绝不能直接交给 SystemExit，
    # 否则 Python 会把 dict 打到 stderr 并以退出码 1 结束，平台会判成"运行失败"。
    return 0 if not isinstance(result, int) else int(result)


def main():
    mode = os.environ.get("HARNESS_MODE", "loop").strip().lower()
    if mode == "serve":
        import serve
        return serve.main(sys.argv[1:])
    if mode == "selftest":
        return selftest()
    return loop(sys.argv[1:])


if __name__ == "__main__":
    _code = main()
    raise SystemExit(_code if isinstance(_code, int) else 0)
