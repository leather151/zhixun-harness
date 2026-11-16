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

SYSTEM_PROMPT = """你是工业模型自进化系统的策略器。输入是一个 JSON 状态，你要输出且只输出一个 JSON 对象：
{"action": "<动作>"}

六个可选动作（只能用这六个词，不能自创）：
- hold：什么都不做，继续观察
- recompute_stats：只重估预处理统计量，不训练
- finetune_1ep：用累积的已标注数据续训 1 个 epoch
- finetune_3ep：用累积的已标注数据续训 3 个 epoch
- swap_candidate：切换到预先训练好的备选模型
- rollback：回滚到上一个被验证更好的检查点

按下面顺序判断，命中即停：
1) 若 caps.epochs_left <= 0 或 caps.updates_left <= 0        -> hold
2) 若 state.candidate_regressed 为 true                      -> rollback
3) 若 state.drift.ks_median >= 0.45 或 指标跌幅达到阈值的 2 倍以上 -> finetune_3ep
4) 若 state.drift.ks_median >= 0.30 或 指标跌幅达到阈值        -> finetune_1ep
5) 若 state.groups 显示误差集中在一两个组（某组指标明显差于其它）-> finetune_1ep
6) 其它情况                                                  -> hold

补充：本系统已有确定性规则兜底，所以宁可保守（拿不准就 hold），也不要输出无法解析的内容。
不要输出解释、不要输出 markdown、不要输出多余字段。

示例：
输入 {"caps": {"epochs_left": 0, "updates_left": 3}, "state": {"drift": {"ks_median": 0.6}}}
输出 {"action": "hold"}

输入 {"caps": {"epochs_left": 5, "updates_left": 3}, "state": {"candidate_regressed": true, "drift": {"ks_median": 0.2}}}
输出 {"action": "rollback"}

输入 {"caps": {"epochs_left": 5, "updates_left": 3}, "state": {"candidate_regressed": false, "drift": {"ks_median": 0.52, "abs_z_median": 0.9}}}
输出 {"action": "finetune_3ep"}
"""


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


def _default_key():
    """取 API Key：环境变量优先，其次工程根目录的 .env.local（该文件已被 gitignore）。"""
    import os
    from pathlib import Path
    for name in ("PYTRIO_API_KEY", "SAMPLER_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    for path in (Path(__file__).resolve().parent.parent / ".env.local",
                 Path("/app/.env.local")):
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if "=" not in line or line.strip().startswith("#"):
                    continue
                k, v = line.split("=", 1)
                if k.strip() in ("PYTRIO_API_KEY", "SAMPLER_API_KEY", "OPENAI_API_KEY"):
                    return v.strip()
    return None


class LLMPolicy:
    name = "llm"

    def __init__(self, endpoint, model, api_key=None, timeout=20.0, cfg=None,
                 enable_thinking=False, max_tokens=128, vote=1):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key = api_key or _default_key()
        self.timeout = timeout
        self.cfg = cfg or {}
        self.fallback = RulePolicy(self.cfg)
        # 实测（PyTRiO 的 Qwen3.5-4B）：推理型模型默认会把 max_tokens 全用在 reasoning 上，
        # content 返回 null；关掉思考后 0.3 s 就给出干净的 {"action": ...}。
        # 但并非所有 OpenAI 兼容服务都认这个字段 → 遇到 400 不支持时自动去掉重试。
        self.enable_thinking = enable_thinking
        self.max_tokens = max_tokens
        # vote>1：用 OpenAI 的 n 参数一次取多个候选做多数投票（小模型提稳的常用手段）
        self.vote = max(1, int(vote))

    # ---------------------------------------------------------------- HTTP
    def _post(self, body):
        import urllib.error
        import urllib.request
        url = self.endpoint if self.endpoint.endswith("/chat/completions") \
            else self.endpoint + "/chat/completions"
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {self.api_key}"}
                                                 if self.api_key else {})})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")[:400]

    def __call__(self, state, caps):
        payload = {
            "model": self.model,
            "temperature": 0 if self.vote <= 1 else 0.6,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(
                    {"state": state, "caps": caps, "allowed_actions": ACTIONS},
                    ensure_ascii=False)},
            ],
        }
        if self.enable_thinking is not None:
            payload["enable_thinking"] = bool(self.enable_thinking)
        if self.vote > 1:
            payload["n"] = self.vote
        try:
            status, text = self._post(json.dumps(payload).encode("utf-8"))
            # 有些服务不认 enable_thinking / n → 逐项去掉再试
            if status == 400:
                changed = False
                if self.enable_thinking is not None and (
                        "enable_thinking" in text or "not supported" in text
                        or "unsupported" in text):
                    payload.pop("enable_thinking", None)
                    self.enable_thinking = None
                    changed = True
                if self.vote > 1 and ("n" in text or "unsupported" in text):
                    payload.pop("n", None)
                    self.vote = 1
                    payload["temperature"] = 0
                    changed = True
                if changed:
                    status, text = self._post(json.dumps(payload).encode("utf-8"))
            if status != 200:
                raise RuntimeError(f"HTTP {status}: {text[:200]}")

            data = json.loads(text)
            actions, raw_all = [], []
            for choice in data.get("choices") or []:
                message = choice.get("message") or {}
                content = (message.get("content") or "").strip()
                if not content:                   # 推理型模型可能把文本放在 reasoning
                    content = (message.get("reasoning")
                               or message.get("reasoning_content") or "").strip()
                raw_all.append(content[:120])
                parsed = _parse_action(content)
                if parsed:
                    actions.append(parsed)
            if not actions:
                raise ValueError(f"模型输出无法解析为动作：{raw_all[:1]}")
            action = max(set(actions), key=actions.count)       # 多数投票
            consensus = actions.count(action)

            if caps.get("updates_left", 0) <= 0 or caps.get("epochs_left", 0) <= 0:
                return {"action": "hold", "reason": "预算已用尽（覆盖 LLM 建议）",
                        "source": self.name, "llm_raw": raw_all[0], "votes": actions}
            if not state["updates_allowed"] and action != "hold":
                return {"action": "hold", "reason": "该模式禁止更新（覆盖 LLM 建议）",
                        "source": self.name, "llm_raw": raw_all[0], "votes": actions}
            return {"action": action,
                    "reason": f"LLM 决策（{consensus}/{len(actions)} 票一致）",
                    "source": self.name, "llm_raw": raw_all[0],
                    "votes": actions, "usage": data.get("usage")}
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


def build(kind, cfg, endpoint=None, model=None, api_key=None,
          default_endpoint=None, default_model=None, enable_thinking=False, vote=1):
    """构造决策器。

    解析顺序（auto 与 llm 一致，避免"显式要 LLM 却静默降级成规则版"）：
      1) 调用方显式给的 endpoint/model/api_key；
      2) 平台注入的环境变量（SAMPLER_* / PYTRIO_* / OPENAI_* …）；
      3) 配置文件里的默认 endpoint/model（本项目默认 PyTRiO 的 Qwen3.8-27B）；
      4) 都没有 → 确定性规则决策器（并在 kind=="llm" 时明确告警，不静默降级）。
    """
    detected = detect_from_env() if kind in ("auto", "llm") else None
    if detected:
        endpoint = endpoint or detected["endpoint"]
        model = model or detected["model"]
        api_key = api_key or detected["api_key"]
    endpoint = endpoint or default_endpoint
    model = model or default_model
    key = api_key or _default_key()
    if endpoint and (key or detected):
        pol = LLMPolicy(endpoint, model or "qwen", api_key=api_key, cfg=cfg,
                        enable_thinking=enable_thinking, vote=vote)
        if detected:
            pol.detected_env = detected["source"]
        elif default_endpoint and endpoint == default_endpoint:
            pol.detected_env = "config.default_endpoint"
        return pol
    if kind == "llm":
        print("[policy] 警告：已要求 LLM 决策，但既没有 endpoint 也没有 API Key"
              "（检查 .env.local / PYTRIO_API_KEY），本轮回落规则版。", flush=True)
    return RulePolicy(cfg)


# 平台可能用这些环境变量注入"驱动 Harness 的大模型"接入信息（按优先级尝试）
_ENV_PAIRS = (
    ("SAMPLER_ENDPOINT", "SAMPLER_MODEL", "SAMPLER_API_KEY"),
    ("HARNESS_LLM_ENDPOINT", "HARNESS_LLM_MODEL", "HARNESS_LLM_KEY"),
    ("PYTRIO_BASE_URL", "PYTRIO_MODEL", "PYTRIO_API_KEY"),
    ("OPENAI_BASE_URL", "OPENAI_MODEL", "OPENAI_API_KEY"),
    ("LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY"),
)


def detect_from_env(environ=None):
    """探测大模型接入信息；找不到返回 None。"""
    import os
    env = environ if environ is not None else os.environ

    def first(*names):
        for n in names:
            v = env.get(n)
            if v and v.strip():
                return v.strip()
        return None

    for ep_key, model_key, key_key in _ENV_PAIRS:
        endpoint = first(ep_key)
        if not endpoint:
            continue
        return {
            "endpoint": endpoint,
            "model": first(model_key) or first("MODEL_NAME") or "qwen",
            "api_key": first(key_key),
            "source": ep_key,
        }
    return None
