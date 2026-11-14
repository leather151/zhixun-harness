"""策略层：把"监测 + 归因"的状态映射成一个动作。

两个实现，接口完全一致：

    policy(state: dict, caps: dict) -> dict(action=..., reason=..., source=...)

1. `RulePolicy`：纯确定性阈值表，**离线、零依赖、100% 可复现**。评测环境里的默认实现。
2. `LLMPolicy`：调用一个 OpenAI 兼容的 chat/completions 接口，让大模型在六个动作里选一个。
   - 只在显式给出 `--llm-endpoint` 时启用；
   - **任何异常（超时/网络/返回不可解析/动作越界）自动回落到 RulePolicy**，并在日志里标 `fallback=true`；
   - 这条设计是为赛题门槛服务的：封闭评测必须"无人干预跑完"，不能把命门交给一个外部模型。
"""
from __future__ import annotations

import json
import urllib.request

ACTIONS = ["hold", "finetune_1ep", "finetune_3ep", "rollback"]

SYSTEM_PROMPT = """你是工业模型自进化系统的策略器。你会收到一个 JSON 状态，必须只输出一个 JSON：
{"action": "<六个动作之一>"}
动作含义：
- hold：什么都不做，继续观察（漂移不显著，或预算已用尽）
- finetune_1ep：用累积的已标注数据小步续训 1 个 epoch（轻度漂移）
- finetune_3ep：用累积的已标注数据续训 3 个 epoch（明显漂移，且上次小步续训不够）
- rollback：回滚到上一个被验证更好的检查点（刚上线的新模型表现变差）
判断原则：先看指标跌幅是否超过噪声，再看输入漂移强度；预算（caps）用尽时一律 hold。
不要输出解释，只输出 JSON。"""


class RulePolicy:
    name = "rule"

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, state, caps):
        c = self.cfg
        if caps.get("updates_left", 0) <= 0 or caps.get("epochs_left", 0) <= 0:
            return {"action": "hold", "reason": "预算已用尽", "source": self.name}
        if not state["updates_allowed"]:
            return {"action": "hold", "reason": "该模式禁止更新（官方 validation 只测不改）",
                    "source": self.name}
        if state.get("failed_updates", 0) >= 1 and state.get("candidate_regressed"):
            return {"action": "rollback", "reason": "上一次上线的新模型在守门窗口上退化",
                    "source": self.name}

        drift_hit = state["drift"]["ks_median"] >= c["ks_median_trigger"] or \
                    state["drift"]["abs_z_median"] >= c["abs_z_median_trigger"]

        if state["primary_higher_is_better"]:
            drop = state["ref_metric"] - state["metric"]
            metric_hit = drop >= c["metric_drop_trigger"]
        else:
            drop = state["metric"] / max(state["ref_metric"], 1e-12) - 1.0
            metric_hit = drop >= c["metric_rise_trigger"]

        if not (drift_hit or metric_hit):
            return {"action": "hold", "reason": "漂移与指标都在阈值内", "source": self.name}

        epochs_left = caps.get("epochs_left", 0)
        if state.get("consecutive_alerts", 0) >= c["escalate_after"] and epochs_left >= 3:
            return {"action": "finetune_3ep",
                    "reason": f"连续 {state['consecutive_alerts']} 窗口报警：漂移/跌幅持续",
                    "source": self.name}
        if epochs_left >= 1:
            return {"action": "finetune_1ep",
                    "reason": "首次报警：先小步续训试一次", "source": self.name}
        return {"action": "hold", "reason": "无可用 epoch 预算", "source": self.name}


class LLMPolicy:
    name = "llm"

    def __init__(self, endpoint, model, api_key=None, timeout=20.0, cfg=None):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.cfg = cfg or {}
        self.fallback = RulePolicy(self.cfg)

    def __call__(self, state, caps):
        body = json.dumps({
            "model": self.model,
            "temperature": 0,
            "max_tokens": 64,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(
                    {"state": state, "caps": caps, "allowed_actions": ACTIONS},
                    ensure_ascii=False)},
            ],
        }).encode("utf-8")
        url = self.endpoint if self.endpoint.endswith("/chat/completions") else \
            self.endpoint + "/chat/completions"
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.api_key}"}
                                                 if self.api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            text = payload["choices"][0]["message"]["content"]
            action = _parse_action(text)
            if action is None:
                raise ValueError(f"模型输出无法解析为动作：{text[:120]}")
            if caps.get("updates_left", 0) <= 0 or caps.get("epochs_left", 0) <= 0:
                return {"action": "hold", "reason": "预算已用尽（覆盖 LLM 建议）",
                        "source": self.name, "llm_raw": text[:200]}
            if not state["updates_allowed"] and action != "hold":
                return {"action": "hold", "reason": "该模式禁止更新（覆盖 LLM 建议）",
                        "source": self.name, "llm_raw": text[:200]}
            return {"action": action, "reason": "LLM 决策", "source": self.name,
                    "llm_raw": text[:200]}
        except Exception as exc:                                    # noqa: BLE001
            out = self.fallback(state, caps)
            out.update({"source": "rule(fallback)", "fallback": True,
                        "fallback_reason": f"{type(exc).__name__}: {exc}"[:200]})
            return out


def _parse_action(text):
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            act = str(obj.get("action", "")).strip()
            return act if act in ACTIONS else None
        except json.JSONDecodeError:
            return None
    for act in ACTIONS:                    # 退一步：找第一个出现的动作名
        if act in text:
            return act
    return None


def build(kind, cfg, endpoint=None, model=None, api_key=None):
    if kind == "llm" and endpoint:
        return LLMPolicy(endpoint, model or "qwen", api_key=api_key, cfg=cfg)
    return RulePolicy(cfg)
