"""LLM / Sampler 接入调试工具：一次调用定位问题。

用途：给定 endpoint / 模型 / key，发一个**真实的诊断状态**过去，打印
  ① 请求体（截断）② 原始响应 ③ 解析出的动作 ④ 耗时 / 错误
这样无论是 404、401、超时、还是"模型输出不是 JSON"，都能一眼看出卡在哪一层。

用法：
    python harness/llm_probe.py --endpoint https://xxx/v1 --model Qwen/Qwen3.5-4B --key trio_xxx
    python harness/llm_probe.py --endpoint http://127.0.0.1:18080/v1 --model mock     # 对 mock 自测
    python harness/llm_probe.py --endpoint ... --repeat 5                              # 测稳定性
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import policy as P      # noqa: E402


def sample_state():
    """构造一个贴近真实闭环的状态（B 包第 2 窗那种：明显漂移 + 预算充足）。"""
    return {
        "state": {
            "window_index": 2, "primary": "rmse", "metric": 135.94, "ref_metric": 125.01,
            "primary_higher_is_better": False,
            "drift": {"ks_median": 0.509, "ks_max": 0.78, "ks_over_0_2": 29,
                      "abs_z_median": 1.43, "abs_z_max": 4.1},
            "top_shifted": [{"feature": "vertical.rms", "z": 2.9, "ks": 0.72},
                            {"feature": "horizontal.band3_energy_6k_12k", "z": 2.1, "ks": 0.66}],
            "consecutive_alerts": 1, "failed_updates": 0, "candidate_regressed": False,
            "updates_allowed": True,
            "groups": {"Bearing1_4": {"n": 122, "rmse": 402.0, "mae": 350.0, "bias": 380.0},
                       "Bearing3_4": {"n": 1515, "rmse": 300.0, "mae": 250.0, "bias": -20.0}},
        },
        "caps": {"updates_left": 3, "epochs_left": 5},
        "allowed_actions": P.ACTIONS,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--key", default="")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args(argv)

    pol = P.LLMPolicy(args.endpoint, args.model, api_key=args.key, timeout=args.timeout,
                      cfg={"ks_median_trigger": 0.25, "abs_z_median_trigger": 0.5,
                           "metric_drop_trigger": 0.02, "metric_rise_trigger": 0.05,
                           "escalate_after": 2})
    payload = sample_state()
    print(f"endpoint = {args.endpoint}")
    print(f"model    = {args.model}")
    print(f"key      = {'*' * 8 + args.key[-4:] if args.key else '(空)'}")
    print(f"状态     = ks_median {payload['state']['drift']['ks_median']}, "
          f"candidate_regressed {payload['state']['candidate_regressed']}, "
          f"caps {payload['caps']}\n")

    ok = 0
    for i in range(1, args.repeat + 1):
        t0 = time.monotonic()
        out = pol(payload["state"], payload["caps"])
        dt = time.monotonic() - t0
        src = out.get("source")
        tag = "OK " if src == "llm" else "回落"
        print(f"[{i}/{args.repeat}] {tag} {dt * 1000:6.0f} ms  action={out.get('action')}  "
              f"source={src}")
        if out.get("llm_raw") is not None:
            raw = out["llm_raw"].replace("\n", " ")
            print(f"        原始响应: {raw[:200]}")
        if out.get("fallback"):
            print(f"        回落原因: {out.get('fallback_reason')}")
        if src == "llm":
            ok += 1

    print(f"\n结果：{ok}/{args.repeat} 次由 LLM 正常决策"
          + ("（其余已自动回落规则版，系统不会因此跑不完）" if ok < args.repeat else ""))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
