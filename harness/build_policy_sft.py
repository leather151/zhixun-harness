"""把闭环真实跑过的决策轨迹整理成 SFT 语料（给 PyTRiO 的 LoRA 微调用）。

数据来源：`runs/<TASK>-<stream>-<时间戳>/windows.jsonl` + `decisions.jsonl`。
每条样本是一个"窗口状态 → 决策动作 + 理由"的监督对：
    输入 = 该窗口的监测/归因结果（指标、参考指标、漂移统计、逐组指标、预算、是否已有失败更新…）
    输出 = 当时真实选的动作（hold / recompute_stats / finetune_1ep / finetune_3ep / rollback）
          + decisions.jsonl 里记录的真实 reason

为什么这样做而不是"造数据"：
- 这些轨迹全部来自我们自己在三个公开数据集上跑出来的闭环，可回溯到 manifest 与 SwanLab run；
- 不虚构任何"教科书式"决策，微调出来的模型就是我们策略层的行为克隆。

输出：`data/sft/policy_sft.jsonl`（每行 {"messages":[{role,content}...]}）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
RUNS = APP / "runs"
OUT = APP / "data" / "sft" / "policy_sft.jsonl"

SYSTEM = (
    "你是工业模型自进化闭环的决策器。每一轮你会收到一个数据流窗口的监测与归因结果，"
    "要在以下动作中选一个并给出简短理由：\n"
    "- hold：不动，继续观察；\n"
    "- recompute_stats：只重算标准化/分位表，不训练；\n"
    "- finetune_1ep：用已标注缓冲轻量续训 1 轮；\n"
    "- finetune_3ep：续训 3 轮（更激进）；\n"
    "- rollback：回滚到上一个被验证更好的检查点。\n"
    "硬规则：updates_allowed=false 时只能 hold（该模式禁止更新）；"
    "updates_left 或 epochs_left 为 0 时不能选 finetune；"
    "candidate_regressed 为真且仍无改善时优先 rollback。\n"
    "只输出一行 JSON：{\"action\": \"<动作>\", \"reason\": \"<不超过 40 字的中文理由>\"}"
)

TASK_CN = {"A": "A 气体传感器六分类（主指标 Macro-F1，越高越好）",
           "B": "B 滚动轴承 RUL 回归（主指标 RMSE，分钟，越低越好）",
           "C": "C 涡扇发动机 RUL 回归（主指标 RMSE，循环，越低越好）"}

ACTIONS = ("hold", "recompute_stats", "finetune_1ep", "finetune_3ep", "rollback")


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compact_state(win, task, stream):
    """把窗口记录压成给模型看的"状态"。不含 action / gate / executed（那些是答案或答案的后果）。"""
    drift = win.get("drift") or {}
    groups = {}
    for name, m in (win.get("by_group") or {}).items():
        groups[name] = {k: round(v, 4) for k, v in m.items() if k in ("n", "rmse", "mae", "bias", "recall")}
    return {
        "task": TASK_CN.get(task, task),
        "stream": stream,
        "window_index": win.get("window_index"),
        "rows": win.get("rows"),
        "primary": win.get("primary"),
        "metric": round(win["metric"], 6) if win.get("metric") is not None else None,
        "ref_metric": round(win["ref_metric"], 6) if win.get("ref_metric") is not None else None,
        "drift": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in drift.items()
                  if k in ("ks_median", "ks_max", "ks_over_0_2", "abs_z_median", "abs_z_max")},
        "top_shifted": drift.get("top_shifted", [])[:3],
        "by_group": groups,
        "updates_used": win.get("updates_used"),
        "epochs_used": win.get("epochs_used"),
    }


def main():
    samples, stats = [], {"runs": 0, "windows": 0, "skipped": 0, "dup": 0}
    seen = set()
    for run_dir in sorted(RUNS.glob("*-*-*")):
        win_path = run_dir / "windows.jsonl"
        if not win_path.is_file():
            continue
        parts = run_dir.name.split("-")
        task = parts[0][:1].upper()
        stream = "train" if "train" in run_dir.name else "official"
        dec_path = run_dir / "decisions.jsonl"
        reasons = {}
        if dec_path.is_file():
            for d in load_jsonl(dec_path):
                idx = d.get("window_index")
                dec = d.get("decision") or {}
                reasons[idx] = dec.get("reason") or ""
        stats["runs"] += 1
        for win in load_jsonl(win_path):
            action = win.get("action")
            if action not in ACTIONS:
                stats["skipped"] += 1
                continue
            state = compact_state(win, task, stream)
            key = json.dumps([state, action], sort_keys=True, ensure_ascii=False)
            if key in seen:
                stats["dup"] += 1
                continue
            seen.add(key)
            reason = reasons.get(win.get("window_index")) or ""
            reason = reason if len(reason) <= 40 else reason[:38] + "…"
            target = json.dumps({"action": action, "reason": reason}, ensure_ascii=False)
            samples.append({"messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(state, ensure_ascii=False, sort_keys=True)},
                {"role": "assistant", "content": target},
            ]})
            stats["windows"] += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    # 分布统计（答辩/报告要用）
    from collections import Counter
    dist = Counter(json.loads(s["messages"][2]["content"])["action"] for s in samples)
    print(f"写出 {OUT}：{len(samples)} 条（{OUT.stat().st_size/1024:.1f} KB）")
    print(f"来源 run 数 {stats['runs']}；窗口 {stats['windows']}；跳过 {stats['skipped']}；去重 {stats['dup']}")
    print("动作分布：", dict(dist))
    lens = sorted(len(s["messages"][1]["content"]) for s in samples)
    print(f"状态长度 中位/最长：{lens[len(lens)//2]} / {lens[-1]} 字符")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
