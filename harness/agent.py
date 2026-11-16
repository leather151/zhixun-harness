"""Agent 层：让 PyTRiO 大模型真正"读历史 → 改配置/代码 → 用工具执行"。

为什么需要这一层：
赛题说明里写的是「**Harness 调用 PyTRiO 大模型来读写代码、使用工具**；评分对象是续训后的
工业模型」。我们原先的 `LLMPolicy` 只是在 6 个固定动作里选一个（hold/finetune/rollback…），
既没有"读实验记录"，也没有"提出代码/超参改动"，更没用工具——这是与要求最明显的差距。

本模块补上这条链路，但**不执行模型写的任意代码**（封闭评测里那样太危险）：

  1. 读历史：把上一轮 `experiments.jsonl` 压成摘要（指标、漂移、试过的配方、成本）；
  2. 提方案：让大模型输出一个**受白名单约束的实验规格 JSON**
     （recipe / epochs / learning_rate / batch_size / rationale），
     这等价于"改训练代码里的超参与配方"，但只能落在允许集合内；
  3. 用工具：把规格当参数调用官方训练入口 `python -m baselines.train`（这是"工具"），
     拿到真实产物 `checkpoint.npz` + `training.json`；
  4. 做决策：候选照旧交给守门（下一窗对比冠军），采纳/驳回/回滚并全部记录。

任何一步失败都退回现有行为（规则版细调），不阻断闭环。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parent.parent

# 白名单：模型只能在下面这些值里挑，避免"改代码"变成不可控行为
RECIPES = ("plain", "class_weight", "distill", "long_average")
BOUNDS = {
    "epochs": (1, 8),
    "learning_rate": (5e-5, 5e-3),
    "batch_size": (64, 256),
}

SPEC_SYSTEM = (
    "你是工业模型自进化系统的实验设计器。你会收到上一轮实验的摘要（历史指标、漂移统计、"
    "已试过的配方与成本）。请据此提出**下一轮实验规格**，只输出一行 JSON：\n"
    '{"recipe": "<plain|class_weight|distill|long_average>", '
    '"epochs": <int 1-8>, "learning_rate": <0.00005-0.005>, '
    '"batch_size": <64|128|256>, "rationale": "<≤40字中文理由>"}\n'
    "约束：epochs 越大越贵；历史显示同类配方已失败时换一个；指标退化时应减小学习率或换配方。"
)


def digest_history(ledger, limit=5):
    """把实验台账压成给模型看的摘要（这就是"读历史"）。"""
    tail = ledger[-limit:] if ledger else []
    rows = []
    for r in tail:
        rows.append({
            "round": r.get("round"),
            "action": (r.get("decision") or {}).get("action"),
            "hypothesis": (r.get("hypothesis") or "")[:40],
            "metric": (r.get("result") or {}).get("metric"),
            "delta": (r.get("result") or {}).get("delta_vs_ref"),
            "drift_ks": (r.get("result") or {}).get("drift_ks_median"),
            "adopted": ((r.get("decision") or {}).get("gate") or {}).get("adopted"),
        })
    tried = [r.get("change") for r in tail if r.get("change")]
    return {"recent": rows, "tried_changes": tried,
            "allowed_recipes": list(RECIPES), "bounds": BOUNDS}


def clamp_spec(spec):
    """把模型给的规格夹进白名单；非法就返回 None（由上层回退）。"""
    if not isinstance(spec, dict):
        return None
    recipe = str(spec.get("recipe") or "").strip()
    if recipe not in RECIPES:
        return None
    out = {"recipe": recipe, "rationale": str(spec.get("rationale") or "")[:60]}
    for key, (lo, hi) in BOUNDS.items():
        raw = spec.get(key)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return None
        if key == "epochs":
            val = int(round(val))
        if key == "batch_size":
            val = int(min(BOUNDS["batch_size"][1], max(BOUNDS["batch_size"][0], round(val / 64) * 64)))
        elif key == "learning_rate":
            val = float(min(hi, max(lo, val)))
        else:
            val = int(min(hi, max(lo, val)))
        out[key] = val
    return out


def propose(policy, task, state, ledger, extra_context=None):
    """让大模型提出实验规格。policy 需支持 `.chat_json(system, user)` 或 `.fallback`。

    返回 (spec, raw_text) ；失败返回 (None, why)。
    """
    digest = digest_history(ledger)
    if extra_context:
        digest["context"] = extra_context
    payload = {"task": task, "state": state, "history": digest}
    try:
        raw = policy.chat_json(SPEC_SYSTEM, json.dumps(payload, ensure_ascii=False))
    except Exception as exc:                                  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"[:200]
    spec = clamp_spec(_first_json(raw))
    return spec, raw[:400]


def _first_json(text):
    if not isinstance(text, str):
        return None
    start = text.find("{")
    while start != -1:
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except Exception:                         # noqa: BLE001
                        break
        start = text.find("{", start + 1)
    return None


def run_tool(task, spec, parent, train_path, validation_path, out_dir, stage_id):
    """用官方训练入口执行该规格（"使用工具"）；返回 (ok, report_or_error, seconds)。"""
    from pathlib import Path as _P
    base = _P(__file__).resolve().parent.parent / "harness" / "tasks.py"
    cmd = [sys.executable, "-m", "baselines.train",
           "--profile", task, "--train", str(train_path), "--validation", str(validation_path),
           "--output", str(out_dir), "--stage-id", stage_id,
           "--recipe", spec["recipe"], "--epochs", str(spec["epochs"]),
           "--learning-rate", str(spec["learning_rate"]),
           "--batch-size", str(spec["batch_size"])]
    if parent:
        cmd += ["--from-baseline", str(parent)]
    env = dict(os.environ, PYTHONPATH=str(APP), PYTHONIOENCODING="utf-8")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", cwd=str(APP), env=env)
    spent = round(time.time() - t0, 1)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "")[-400:], spent
    tj = Path(out_dir) / "training.json"
    if not tj.is_file():
        return False, "training.json missing", spent
    return True, json.loads(tj.read_text(encoding="utf-8")), spent
