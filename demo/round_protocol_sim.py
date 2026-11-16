"""本地复刻平台的"多轮候选 / 采纳"协议 —— 不消耗任何平台试运行次数。

平台每轮做四件事：
  1. 把**当前被选中的模型**交给我们（挂在 /previous）；
  2. 我们的容器跑一轮闭环，把产物写到 /model/<TASK>/checkpoint.npz；
  3. 平台用公开验证集评这个候选；
  4. 候选优于当前模型 → 采用（显示"采用本轮模型"），否则保留（"候选未采用，保留此前模型"）。

这个脚本把上面四步在本机跑一遍，输出一张与平台同样口径的表格，用于：
  - 在不消耗试运行的前提下确认"交付链路通不通"；
  - 演示视频里展示"闭环能真的把模型一轮轮推好"。

注意：**这不是平台成绩**。平台用的是隐藏测试流，本地拿不到（规则也禁止获取）。
本地表格只反映"在公开验证集上的采纳行为"。

用法：
    python demo/round_protocol_sim.py --task B --rounds 3
    python demo/round_protocol_sim.py --task A --rounds 3 --windows 2
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP / "harness"))

import metrics as M          # noqa: E402
import tasks as T            # noqa: E402


def load_vendor(task):
    import importlib.util
    name = f"_sim_vendor_{task}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, T.VENDOR_ROOT / task / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def evaluate(task, ckpt):
    rows = T.load_jsonl(T.task_base(task) / "validation.jsonl", T.TASKS[task]["classes"] or None)
    model = load_vendor(task).Model.load(ckpt)
    pred = model.predict(rows["x"])
    if T.TASKS[task]["kind"] == "classification":
        p = pred.argmax(1)
        return {"macro_f1": M.macro_f1(rows["y"].astype(int), p, len(T.TASKS[task]["classes"])),
                "accuracy": float((p == rows["y"]).mean())}
    p = pred.ravel()
    return {"rmse": M.rmse(p, rows["y"]), "mae": M.mae(p, rows["y"]), "bias": M.bias(p, rows["y"])}


def primary_of(task):
    return T.TASKS[task]["primary"]


def better(task, cand, incumbent):
    key = primary_of(task)
    if T.TASKS[task]["higher_is_better"]:
        return cand[key] > incumbent[key]
    return cand[key] < incumbent[key]


def fmt(task, met):
    if T.TASKS[task]["kind"] == "classification":
        return f"Macro-F1 {met['macro_f1']:.6f} / Acc {met['accuracy']:.6f}"
    return f"RMSE {met['rmse']:.6f} / MAE {met['mae']:.6f}"


def run_round(task, parent_dir, out_dir, windows, policy):
    """跑一轮我们自己的闭环（等价于平台调一次容器）。"""
    cmd = [sys.executable, "-u", str(APP / "harness" / "entry.py"),
           "--task", task, "--stream", "train", "--windows", str(windows),
           "--policy", policy, "--swan", "off",
           "--previous", str(parent_dir), "--model-out", str(out_dir / "model"),
           "--out", str(out_dir / "runs")]
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    spent = time.time() - t0
    delivered = out_dir / "model" / task / "checkpoint.npz"
    actions = []
    for line in (proc.stdout or "").splitlines():
        if line.startswith(f"[{task}] w"):
            actions.append(line.split("->")[-1].strip())
    return {"returncode": proc.returncode, "seconds": round(spent, 1),
            "checkpoint": delivered if delivered.is_file() else None,
            "actions": actions, "tail": (proc.stdout or "").strip().splitlines()[-3:]}


def main(argv=None):
    ap = argparse.ArgumentParser(description="本地复刻平台多轮候选/采纳协议")
    ap.add_argument("--task", default="B", choices=list(T.TASKS))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--windows", type=int, default=2, help="每轮闭环的数据流窗口数")
    ap.add_argument("--policy", default="rule")
    ap.add_argument("--keep", action="store_true", help="保留临时目录（调试用）")
    args = ap.parse_args(argv)

    task = args.task
    work = Path(tempfile.mkdtemp(prefix=f"sim-{task}-"))
    # 第 0 轮：平台的父模型（用官方起点等价模拟）
    parent_dir = work / "parent"
    parent_dir.mkdir(parents=True)
    shutil.copy2(T.MODEL_ROOT / task / "checkpoint_official_baseline.npz",
                 parent_dir / "checkpoint.npz")
    selected = parent_dir / "checkpoint.npz"
    incumbent = evaluate(task, selected)

    print(f"任务 {task}｜主指标 {primary_of(task)}｜每轮窗口 {args.windows}｜共 {args.rounds} 轮")
    print(f"父模型（平台起点）：{fmt(task, incumbent)}")
    print()
    header = f"{'轮次':<5}{'候选公开验证指标':<46}{'模型选择':<26}耗时"
    print(header)
    print("-" * len(header) * 2)
    adopted = 0
    rows = []
    for i in range(1, args.rounds + 1):
        out = work / f"round-{i:02d}"
        attempt = run_round(task, parent_dir, out, args.windows, args.policy)
        if attempt["returncode"] != 0 or attempt["checkpoint"] is None:
            print(f"{i:<5}{'运行失败':<46}{'保留此前模型':<26}{attempt['seconds']}s")
            rows.append({"round": i, "ok": False})
            continue
        cand = evaluate(task, attempt["checkpoint"])
        take = better(task, cand, incumbent)
        label = "采用本轮模型" if take else "候选未采用，保留此前模型"
        print(f"{i:<5}{fmt(task, cand):<46}{label:<26}{attempt['seconds']}s")
        print(f"     {'':<3}本轮动作：{' / '.join(attempt['actions']) or '（无更新）'}")
        rows.append({"round": i, "ok": True, "candidate": cand,
                     "incumbent": incumbent, "adopted": take})
        if take:
            adopted += 1
            incumbent = cand
            # 平台会把被选中的模型作为下一轮的父模型
            shutil.rmtree(parent_dir, ignore_errors=True)
            parent_dir = work / f"parent-{i:02d}"
            parent_dir.mkdir(parents=True)
            shutil.copy2(attempt["checkpoint"], parent_dir / "checkpoint.npz")
            selected = parent_dir / "checkpoint.npz"

    print()
    print(f"最终采用：第 {args.rounds} 轮后的当前模型（共采纳 {adopted} 次）")
    print(f"最终公开验证指标：{fmt(task, incumbent)}")
    print(f"父模型指标      ：{fmt(task, evaluate(task, T.MODEL_ROOT / task / 'checkpoint_official_baseline.npz'))}")
    print()
    print("说明：以上是本地公开验证集上的采纳行为，**不是平台成绩**（隐藏测试流本地不可得）。")
    report = {"task": task, "rounds": rows, "adopted": adopted,
              "final": incumbent, "workdir": str(work)}
    out_json = APP / "runs" / f"sim_rounds_{task}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[json] {out_json}")
    if not args.keep:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
