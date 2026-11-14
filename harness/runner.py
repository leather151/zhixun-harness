"""智训引擎 · 工业模型自进化闭环（监测 → 归因 → 决策 → 优化 → 守门 → 回滚）

用法（本地，官方数据）：
    python harness/runner.py --task B --stream official --windows 3
    python harness/runner.py --task B --stream train    --windows 4      # 可更新（train 内部切分）
    python harness/runner.py --task A --stream train --policy llm \
        --llm-endpoint http://127.0.0.1:8000/v1 --llm-model qwen

容器里（封闭评测，无网络、无 SwanLab）：
    python /app/harness/runner.py --task all --stream auto --swan off

合规要点（写进日志，可审计）：
- `--stream official` 的窗口来自官方 validation.jsonl，**只测不改**，更新动作被强制禁用；
- 更新只使用官方 train.jsonl 内部切分出的数据（`--stream train`）；
- 守门（gate）用"下一窗"做冠军/挑战者对比，不用官方 validation 做模型选择。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import metrics as M          # noqa: E402
import policy as P           # noqa: E402
import runlog as L           # noqa: E402
import tasks as T            # noqa: E402

PRIMARY_NAME = {"macro_f1": "Macro-F1", "rmse": "RMSE"}
# 数据集短名：SwanLab 的实验名要一眼看出是哪个数据集（不是一个 gas 项目里混着跑）
DATASET_TAG = {"A": "gas", "B": "bearing", "C": "cmapss"}


# ------------------------------------------------------------------ 工具

def subsample(rows, limit):
    n = len(rows["y"])
    if n <= limit:
        return rows
    idx = np.linspace(0, n - 1, limit).astype(int)
    return T.take(rows, list(idx))


def fmt_metric(name, value):
    return f"{value:.4f}" if name in ("macro_f1", "accuracy") else f"{value:.2f}"


def better(spec, cand_value, base_value, margin=0.0):
    if spec["higher_is_better"]:
        return cand_value >= base_value + margin
    return cand_value <= base_value - margin


# ------------------------------------------------------------------ 主循环

def run(task, args, cfg):
    spec = T.TASKS[task]
    primary = spec["primary"]
    mode = args.stream
    if mode == "auto":
        root = Path(args.data_root) if args.data_root else T.DATA_ROOT
        mode = "official" if (root / spec["dir"] / "validation.jsonl").exists() else "train"
    stream = T.build_stream(task, mode, args.windows, args.data_root)
    windows = stream["windows"]
    history = stream["history"]
    ref = stream["reference"]
    names = T.feature_names(history["paths"])

    outdir = Path(args.out) / f"{task}-{mode}-{L.stamp()}"
    logger = L.RunLogger(outdir, swan=args.swan,
                         experiment=f"{task}-{DATASET_TAG[task]}-{mode}-{args.policy}",
                         project=args.swan_project, mode=args.swan_mode,
                         config={"task": task, "dataset": DATASET_TAG[task], "stream": mode,
                                 "policy": args.policy, "windows": len(windows),
                                 "primary": primary, "caps": cfg["caps"],
                                 "policy_cfg": cfg["policy"]})

    start_ckpt = Path(args.checkpoint) if args.checkpoint else T.checkpoint_path(task)
    if not start_ckpt.exists():
        raise FileNotFoundError(f"起点检查点不存在：{start_ckpt}")

    # 漂移参照：起点模型在"历史数据"上的预测分布（抽样子集，保证大包也快）
    base_model = T.load_model(task, start_ckpt)
    ref_sub = subsample(ref, 20000)
    ref_x = ref_sub["x"]
    ref_pred = base_model.predict(subsample(ref, 4000)["x"])

    ckpt_stack = [str(start_ckpt)]
    current = str(start_ckpt)
    pending = None                # {"ckpt":..., "action":..., "at":...}
    updates_used = epochs_used = 0
    training_seconds = 0.0
    window_metrics, previous = [], []
    consecutive_alerts = failed_updates = 0
    candidate_regressed = False
    events = []
    t_start = time.monotonic()
    caps = dict(cfg["caps"])
    caps["max_windows"] = min(caps["max_windows"], len(windows))

    # 更新用的"已标注缓冲"落盘位置（只装 train 内部的行）
    buffer_path = outdir / "buffer.jsonl"
    official_val = Path(stream["base"]) / "validation.jsonl"

    for index, window in enumerate(windows):
        if index >= caps["max_windows"]:
            break
        if updates_used > 0 and (time.monotonic() - t_start) > caps["max_seconds"]:
            events.append({"window": index, "event": "stop", "reason": "超过 max_seconds 上限"})
            break

        # ---- 1. 当前模型的窗口表现 ----
        model = T.load_model(task, current)
        ev = T.evaluate(spec, model, window)
        value = ev["metrics"][primary]

        # ---- 2. 守门：上一窗训练出的挑战者，在这一窗和冠军对比 ----
        gate = None
        if pending is not None:
            ev_cand = T.evaluate(spec, T.load_model(task, pending["ckpt"]), window)
            cand_value = ev_cand["metrics"][primary]
            promote = better(spec, cand_value, value, cfg["gate"]["margin"])
            gate = {"window": index, "candidate": str(pending["ckpt"]),
                    "candidate_metric": cand_value, "incumbent_metric": value,
                    "promoted": bool(promote)}
            if promote:
                ckpt_stack.append(pending["ckpt"])
                current = pending["ckpt"]
                ev = ev_cand
                value = cand_value
                failed_updates = 0
                candidate_regressed = False
            else:
                failed_updates += 1
                candidate_regressed = True
            events.append({"window": index, "event": "gate", **gate})
            pending = None

        window_metrics.append(value)

        # ---- 3. 监测 + 归因 ----
        drift = M.drift_report(ref_x, window["x"], ref_pred, ev["pred"], names)
        ref_value = float(np.mean(previous)) if previous else value

        # ---- 4. 决策 ----
        caps_live = {"updates_left": caps["max_updates"] - updates_used,
                     "epochs_left": caps["max_epochs"] - epochs_used}
        state = {
            "window_index": index,
            "primary": primary,
            "metric": value,
            "ref_metric": ref_value,
            "primary_higher_is_better": spec["higher_is_better"],
            "drift": {k: drift[k] for k in ("ks_median", "ks_max", "ks_over_0_2",
                                            "abs_z_median", "abs_z_max")},
            "top_shifted": drift["top_shifted"][:3],
            "consecutive_alerts": consecutive_alerts,
            "failed_updates": failed_updates,
            "candidate_regressed": candidate_regressed,
            "updates_allowed": bool(stream["updates_allowed"]),
            "groups": ev["by_group"],
        }
        decision = args._policy(state, caps_live)
        action = decision["action"]
        events.append({"window": index, "event": "decision", **decision})
        logger.log_decision({"window_index": index, "action": action, "decision": decision,
                             "state": {"metric": value, "ref_metric": ref_value,
                                       "drift": state["drift"]}})

        # ---- 5. 执行 ----
        executed = None
        if action == "rollback":
            if len(ckpt_stack) > 1:
                ckpt_stack.pop()
                current = ckpt_stack[-1]
                executed = {"action": "rollback", "to": current}
                candidate_regressed = False
            else:
                executed = {"action": "rollback", "skipped": "已在最初始检查点"}
        elif action.startswith("finetune"):
            epochs = cfg["finetune"]["epochs_light"] if action == "finetune_1ep" else \
                cfg["finetune"]["epochs_heavy"]
            epochs = int(min(epochs, caps_live["epochs_left"]))
            if epochs <= 0:
                executed = {"action": action, "skipped": "epoch 预算用尽"}
            else:
                T.write_buffer(buffer_path, history, *windows[:index + 1],
                               table=T.model_table(T.load_model(task, current)))
                cand_dir = outdir / f"cand-w{index + 1}-{action}"
                t0 = time.monotonic()
                report, cand_path = T.fine_tune(
                    task, parent=current, train_path=buffer_path, validation_path=official_val,
                    out_dir=cand_dir, epochs=epochs,
                    learning_rate=cfg["finetune"]["learning_rate"],
                    stage_id=f"harness-w{index + 1}")
                spent = time.monotonic() - t0
                training_seconds += spent
                updates_used += 1
                epochs_used += epochs
                pending = {"ckpt": str(cand_path / "checkpoint.npz"), "action": action,
                           "at": index}
                executed = {"action": action, "epochs": epochs, "seconds": round(spent, 3),
                            "steps": [report["initial_steps"], report["final_steps"]],
                            "parent_sha256": report["parent_checkpoint_sha256"],
                            "candidate_sha256": report["checkpoint_sha256"],
                            "candidate_val": report["validation_after"]}
        if executed:
            events.append({"window": index, "event": "execute", **executed})

        if action != "hold" and stream["updates_allowed"]:
            consecutive_alerts += 1
        elif action == "hold":
            consecutive_alerts = 0

        # ---- 6. 落盘 ----
        payload = {
            "window_index": index,
            "window": window["name"],
            "rows": int(len(window["y"])),
            "live_checkpoint": Path(current).parent.name or Path(current).name,
            "primary": primary,
            "metric": value,
            "ref_metric": ref_value,
            "metrics": ev["metrics"],
            "by_group": ev["by_group"],
            "drift": drift,
            "action": action,
            "action_source": decision.get("source"),
            "gate": gate,
            "executed": executed,
            "updates_used": updates_used,
            "epochs_used": epochs_used,
        }
        logger.log_window(payload, metrics={
            **ev["metrics"],
            "drift_ks_median": drift["ks_median"],
            "drift_abs_z_median": drift["abs_z_median"],
            "drift_features_over_0_2": drift["ks_over_0_2"],
            "updates_used": updates_used,
            "epochs_used": epochs_used,
            "training_seconds": round(training_seconds, 3),
            "promoted": int(bool(gate and gate["promoted"])),
            "rows": int(len(window["y"])),
        })
        previous.append(value)
        print(f"[{task}] w{index + 1}/{len(windows)}  {primary}={fmt_metric(primary, value)}  "
              f"(ref {fmt_metric(primary, ref_value)})  drift ks_med={drift['ks_median']:.3f} "
              f"|z|med={drift['abs_z_median']:.2f}  -> {action}"
              + (f"  [promoted]" if gate and gate["promoted"] else "")
              + (f"  [{executed['action']}:{executed.get('epochs', '')}ep]"
                 if executed and executed.get("epochs") else ""))

    # ------------------------------------------------------------------ 汇总
    # 交付产物：把最终工业模型落到固定路径 <out>/model/<TASK>/checkpoint.npz，
    # 平台评测就拿它；manifest.json 记录来源、指标与成本，便于审计。
    import hashlib
    import shutil
    model_dir = Path(args.out) / "model" / task
    model_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = model_dir / "checkpoint.npz"
    if Path(current).resolve() != final_ckpt.resolve():
        shutil.copy2(current, final_ckpt)
    manifest = {
        "task": task,
        "checkpoint": str(final_ckpt),
        "sha256": "sha256:" + hashlib.sha256(final_ckpt.read_bytes()).hexdigest(),
        "parent_checkpoint": str(ckpt_stack[0]),
        "stream_mode": mode,
        "policy": args.policy,
        "primary": primary,
        "metric_series": window_metrics,
        "promotions": sum(1 for e in events
                          if e.get("event") == "gate" and e.get("promoted")),
        "updates_used": updates_used,
        "epochs_used": epochs_used,
        "training_seconds": round(training_seconds, 3),
        "note": "由 harness 闭环产出的最终工业模型；逐轮证据见同目录 windows.jsonl / decisions.jsonl",
    }
    (model_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "task": task,
        "stream_mode": mode,
        "stream_note": stream["note"],
        "policy": args.policy,
        "policy_impl": getattr(args._policy, "name", "?"),
        "policy_detected_env": getattr(args._policy, "detected_env", None),
        "windows": len(window_metrics),
        "primary": primary,
        "metric_first": window_metrics[0] if window_metrics else None,
        "metric_last": window_metrics[-1] if window_metrics else None,
        "metric_best": (max(window_metrics) if spec["higher_is_better"] else
                        min(window_metrics)) if window_metrics else None,
        "metric_series": window_metrics,
        "updates_used": updates_used,
        "epochs_used": epochs_used,
        "training_seconds": round(training_seconds, 3),
        "promotions": sum(1 for e in events if e.get("event") == "gate" and e.get("promoted")),
        "rollbacks": sum(1 for e in events if e.get("event") == "execute"
                         and e.get("action") == "rollback"),
        "start_checkpoint": str(start_ckpt),
        "final_checkpoint": current,
        "artifact": str(final_ckpt),
        "artifact_sha256": manifest["sha256"],
        "caps": caps,
        "events": events,
        "swan": {"enabled": logger.swan is not None, "error": logger.swan_error,
                 "project": logger.swan_project, "experiment": logger.swan_experiment,
                 "mode": getattr(logger, "swan_mode", None), "url": logger.swan_url},
        "cost_per_point": None,
    }
    if len(window_metrics) > 1 and updates_used:
        gain = (window_metrics[-1] - window_metrics[0]) if spec["higher_is_better"] else \
            (window_metrics[0] - window_metrics[-1])
        if gain:
            summary["cost_per_point"] = round(training_seconds / abs(gain), 3)

    logger.summary(summary)
    _write_report(outdir / "report.md", task, spec, windows, summary, primary)
    logger.close()
    print(f"\n[{task}] 完成：{outdir}")
    print(f"  指标序列 {primary}: {', '.join(fmt_metric(primary, v) for v in window_metrics)}")
    print(f"  上线 {summary['promotions']} 次 / 回滚 {summary['rollbacks']} 次 / "
          f"训练 {training_seconds:.2f} s / {epochs_used} epoch")
    swan = summary["swan"]
    if swan["enabled"]:
        print(f"  SwanLab：{swan['project']} / {swan['experiment']}（{swan['mode']}）"
              + (f"\n  记录链接：{swan['url']}" if swan["url"] else "\n  （未取到 URL，登录后可 swanlab 网页查看）"))
    else:
        print(f"  SwanLab：未记录（{swan['error'] or '已关闭'}）")
    return summary


def _write_report(path, task, spec, windows, summary, primary):
    rows = summary["metric_series"]
    lines = [
        f"# 自进化闭环运行报告 · 任务 {task}", "",
        f"- 数据流模式：`{summary['stream_mode']}` —— {summary['stream_note']}",
        f"- 策略：`{summary['policy']}`；主指标：{PRIMARY_NAME.get(primary, primary)}"
        f"（{'越高越好' if spec['higher_is_better'] else '越低越好'}）",
        f"- 窗口数：{summary['windows']}；上线 {summary['promotions']} 次；"
        f"回滚 {summary['rollbacks']} 次",
        f"- 训练开销：{summary['training_seconds']} s / {summary['epochs_used']} epoch / "
        f"{summary['updates_used']} 次更新",
        f"- 起点检查点：`{summary['start_checkpoint']}`",
        f"- 最终检查点：`{summary['final_checkpoint']}`",
        f"- SwanLab：{'已记录' if summary['swan']['enabled'] else '未启用'}"
        f"{('（原因：' + summary['swan']['error'] + '）') if summary['swan']['error'] else ''}",
        "", "## 逐窗口指标", "",
        "| 窗口 | 样本数 | " + PRIMARY_NAME.get(primary, primary) + " | 决策 | 说明 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for index, value in enumerate(rows):
        decision = next((e for e in summary["events"]
                         if e.get("event") == "decision" and e.get("window") == index), {})
        gate = next((e for e in summary["events"]
                     if e.get("event") == "gate" and e.get("window") == index), None)
        note = []
        if gate:
            note.append("守门：" + ("通过" if gate["promoted"] else "驳回"))
        if decision:
            note.append(str(decision.get("reason", ""))[:40])
        lines.append(f"| {index + 1} | {len(windows[index]['y'])} | "
                     f"{fmt_metric(primary, value)} | {decision.get('action', '')} | "
                     f"{'；'.join(note)} |")
    lines += ["", "## 事件流水", "", "```json",
              json.dumps(summary["events"], ensure_ascii=False, indent=1), "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------------ CLI

def parse(argv=None):
    ap = argparse.ArgumentParser(description="工业模型自进化闭环")
    ap.add_argument("--task", default="B", help="A / B / C（或 all）")
    ap.add_argument("--stream", default="official", choices=["official", "train", "auto"],
                    help="official=官方验证集只测不改；train=train 内部切分（可更新）；auto=按数据自动选")
    ap.add_argument("--windows", type=int, default=4, help="把数据流切成几个窗口")
    ap.add_argument("--policy", default="auto", choices=["auto", "rule", "llm"],
                    help="auto=有 LLM 接入就用 LLM（失败回落规则），没有就用规则；rule=纯规则；llm=强制 LLM")
    ap.add_argument("--llm-endpoint", default=None)
    ap.add_argument("--llm-model", default=None)
    ap.add_argument("--llm-key", default=None)
    ap.add_argument("--checkpoint", default=None, help="覆盖起点检查点")
    ap.add_argument("--data-root", default=None, help="官方数据根目录（含 A_气体传感器/...）")
    ap.add_argument("--out", default=str(HERE.parent / "runs"))
    ap.add_argument("--swan", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--swan-project", default=None,
                    help="SwanLab 项目名（默认 zhixun-engine-3tasks，可用 SWANLAB_PROJECT 覆盖）")
    ap.add_argument("--swan-mode", default=None, choices=["cloud", "local", "disabled"],
                    help="默认 cloud（要拿 HTTPS 记录链接）；容器内用 off")
    ap.add_argument("--config", default=str(HERE / "config.json"))
    return ap.parse_args(argv)


def _argv_for(args, task):
    """把 --task all 拆成每个任务一个子进程（vendor 里的 model/train 是顶层模块，不能混用）。"""
    argv = [sys.executable, "-u", str(Path(__file__).resolve()), "--task", task,
            "--stream", args.stream, "--windows", str(args.windows),
            "--policy", args.policy, "--out", args.out, "--swan", args.swan,
            "--config", args.config]
    if args.data_root:
        argv += ["--data-root", args.data_root]
    if args.policy == "llm":
        for flag, value in (("--llm-endpoint", args.llm_endpoint),
                            ("--llm-model", args.llm_model), ("--llm-key", args.llm_key)):
            if value:
                argv += [flag, value]
    if args.checkpoint and len(T.TASKS) == 1:
        argv += ["--checkpoint", args.checkpoint]
    return argv


def main(argv=None):
    args = parse(argv)
    if args.task == "all":
        import subprocess
        codes = {}
        for task in T.TASKS:
            print(f"\n{'=' * 64}\n  task {task}\n{'=' * 64}", flush=True)
            codes[task] = subprocess.run(_argv_for(args, task)).returncode
        bad = {k: v for k, v in codes.items() if v != 0}
        print(f"\n全部任务结束：{codes}")
        if bad:
            raise SystemExit(f"以下任务失败：{bad}")
        return codes
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    args._policy = P.build(args.policy, cfg["policy"], args.llm_endpoint,
                           args.llm_model, args.llm_key)
    return run(args.task, args, cfg)


if __name__ == "__main__":
    main()
