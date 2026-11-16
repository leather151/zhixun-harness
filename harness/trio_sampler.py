"""路线二：用 PyTRiO SDK 直接采样**我们自己微调的 sampler 权重**（离线可用）。

背景：
- 评测环境断网，只能用平台提供的 PyTRiO 服务。
- 平台若只暴露"host gateway RPC"，我们能通过 `HARNESS_LLM_CMD`（stdin/stdout JSON）接上；
- 但若要用**我们微调出来的权重** `trio://run_.../sampler_weights/...`，就必须用 PyTRiO SDK 的
  `create_sampling_client(model_path=...)` —— 这一步和 OpenAI 兼容的 chat 接口不是一回事。

因此镜像里做两件事（互不干扰）：
1. **主运行时保持原样**（python3.12-slim + numpy 1.26.4，已验证的那套）；
2. 另建 `/app/.trio-venv` 装 pytrio + transformers + modelscope（**离线 wheelhouse 安装**），
   只有本模块需要时才调用它。

为什么用子进程调 venv：避免 transformers/modelscope 对 numpy 的要求（2.x）污染主运行时，
也避免 import 失败拖垮整个闭环。任何一步失败都会明确返回失败原因，上层回落 HTTP → 规则版。

用法（作为命令行）：
    /app/.trio-venv/bin/python /app/harness/trio_sampler.py --model-path trio://... --prompt-file req.json
        stdout: {"content": "...", "model_used": "trio://...", "usage": {...}}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

VENV_PYTHON = os.environ.get("HARNESS_TRIO_VENV_PYTHON", "/app/.trio-venv/bin/python")
CACHE_DIR = os.environ.get("HARNESS_TRIO_CACHE", "/opt/trio-cache")
BASE_MODEL = os.environ.get("HARNESS_TRIO_BASE_MODEL", "Qwen/Qwen3.5-4B")


def available():
    """SDK 是否可用（在没有 pytrio 的主运行时里也能安全询问）。"""
    return Path(VENV_PYTHON).is_file()


def sample_via_sdk(model_path, messages, max_tokens=128, temperature=0.0, seed=0,
                   base_model=BASE_MODEL, timeout=120):
    """用 SDK 采样。**必须在装有 pytrio 的解释器里调用**（通常是 /app/.trio-venv/bin/python）。"""
    import subprocess
    import tempfile

    payload = {"model_path": model_path, "base_model": base_model, "messages": messages,
               "max_tokens": int(max_tokens), "temperature": float(temperature),
               "seed": int(seed), "cache_dir": CACHE_DIR}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
        req_path = fh.name
    try:
        proc = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker",
                               "--request", req_path],
                              capture_output=True, timeout=timeout)
        out = (proc.stdout or b"").decode("utf-8", "replace").strip()
        if proc.returncode != 0:
            raise RuntimeError(f"sdk worker exit {proc.returncode}: "
                               f"{(proc.stderr or b'')[:300].decode('utf-8', 'replace')}")
        return json.loads(out)
    finally:
        try:
            os.unlink(req_path)
        except OSError:
            pass


def _worker(request_path):
    """在装有 pytrio 的解释器里真正执行采样。"""
    import pytrio as trio

    req = json.loads(Path(request_path).read_text(encoding="utf-8"))
    client = trio.ServiceClient()
    try:
        sc = client.create_sampling_client(base_model=req["base_model"],
                                           model_path=req["model_path"])
    except TypeError:
        # 老版本 SDK 不支持 model_path 关键字 → 明确报出来，不要静默用错模型
        raise SystemExit("this pytrio SDK does not accept model_path; "
                         "cannot load a fine-tuned sampler weight")
    tokenizer = sc.get_tokenizer(cache_dir=req.get("cache_dir"))
    prompt = tokenizer.apply_chat_template(req["messages"], tokenize=False,
                                           add_generation_prompt=True)
    ids = tokenizer.encode(prompt)
    import pytrio as _t
    params = _t.SamplingParams(max_tokens=req["max_tokens"], temperature=req["temperature"],
                               seed=req["seed"])
    resp = sc.sample(prompt=_t.ModelInput.from_ints(ids), num_samples=1,
                     sampling_params=params).result()
    text = (resp.sequences[0].text or "").strip()
    print(json.dumps({"content": text, "model_used": req["model_path"],
                      "base_model": req["base_model"]}, ensure_ascii=False))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--worker", action="store_true", help="内部：在装有 pytrio 的解释器里跑")
    ap.add_argument("--request", help="请求 JSON 路径（worker 用）")
    ap.add_argument("--model-path", help="trio://run_.../sampler_weights/...")
    ap.add_argument("--prompt-file", help="包含 {\"messages\": [...]} 的 JSON")
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args(argv)

    if args.worker:
        return _worker(args.request)

    messages = json.loads(Path(args.prompt_file).read_text(encoding="utf-8"))["messages"]
    print(json.dumps(sample_via_sdk(args.model_path, messages, max_tokens=args.max_tokens),
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
