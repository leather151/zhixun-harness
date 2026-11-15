"""本地 mock 的 OpenAI 兼容 LLM 服务，用来端到端验证 Harness 的 LLM 决策链路。

它模拟 PyTRiO 的 /v1/chat/completions：
  - 收 OpenAI 标准请求体；
  - 把 user 消息里的状态 JSON 解析出来；
  - 用一套"伪 LLM"规则在六个动作里选一个，按 {"action": ...} 返回（正是我们要求的输出格式）。

这样可以在没有 PyTRiO 真实地址的情况下，验证：
  请求构造 → 响应解析 → 动作校验 → 预算覆盖 → 失败回落 这一整条链路。

用法：
    python harness/_mock_llm.py 18080            # 起服务
    python harness/runner.py --task B --stream train --windows 3 \
        --policy llm --llm-endpoint http://127.0.0.1:18080/v1 --llm-model Qwen/Qwen3.5-4B --llm-key test
"""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

CALLS = {"n": 0}


def decide(state, caps):
    """伪 LLM 的"推理"：看指标跌幅与漂移量决定动作（模拟一个训练过的策略器）。"""
    drift = state.get("drift", {})
    ks = drift.get("ks_median", 0.0)
    z = drift.get("abs_z_median", 0.0)
    if caps.get("epochs_left", 0) <= 0 or caps.get("updates_left", 0) <= 0:
        return "hold", "预算用尽，保守不动"
    if state.get("candidate_regressed"):
        return "rollback", "上一版在守门窗口退化，回滚"
    if ks >= 0.45 and z >= 0.8:
        return "finetune_3ep", "输入漂移显著，需要更大步长"
    if ks >= 0.30 or z >= 0.50:
        return "finetune_1ep", "轻度漂移，小步续训"
    return "hold", "漂移在阈值内"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):     # 静默
        pass

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            self.send_error(404, "no route")
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw)
            user_msg = next(m["content"] for m in body.get("messages", [])
                            if m.get("role") == "user")
            payload = json.loads(user_msg)
            state, caps = payload["state"], payload["caps"]
            action, reason = decide(state, caps)
            CALLS["n"] += 1
            content = json.dumps({"action": action}, ensure_ascii=False)
            print(f"[mock] call#{CALLS['n']} model={body.get('model')} "
                  f"ks={state.get('drift', {}).get('ks_median'):.3f} "
                  f"metric={state.get('metric')} -> {action}（{reason}）", flush=True)
        except Exception as exc:                                # noqa: BLE001
            content = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
            print(f"[mock] 解析失败：{exc}", flush=True)

        out = json.dumps({
            "id": "chatcmpl-mock", "object": "chat.completion", "model": "mock-qwen",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        out = b'{"object":"list","data":[{"id":"Qwen/Qwen3.5-4B"}]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
    print(f"mock LLM 监听 http://127.0.0.1:{port}/v1/chat/completions", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
