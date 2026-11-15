"""验证 PyTRiO 侧产物：1) 真实 loss 字段长什么样；2) 用保存的 Sampler 权重做推理，看行为对不对。

为什么要验：`trio_sft_policy.py` 里我把没解析到的 loss 记成了 nan。要区分两种情况：
  (a) 只是我没取对字段，训练其实正常；
  (b) 训练真的发散/空转，那这个权重就不能拿去交。
判据用"行为"而不是"loss 数字"：让微调后的模型对留出的窗口照原样输出动作 JSON，
能和真实决策对上才叫训上了。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
SFT = APP / "data" / "sft" / "policy_sft.jsonl"
REPORT = APP / "runs" / "trio_sft_report.json"


def load_key():
    if os.environ.get("PYTRIO_API_KEY"):
        return os.environ["PYTRIO_API_KEY"]
    for line in (APP / ".env.local").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("PYTRIO_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("找不到 PYTRIO_API_KEY")


os.environ["PYTRIO_API_KEY"] = load_key()
import pytrio as trio  # noqa: E402

report = json.loads(REPORT.read_text(encoding="utf-8"))
path = report["sampler_path"]
samples = [json.loads(l) for l in SFT.read_text(encoding="utf-8").splitlines() if l.strip()]
print(f"[setup] sampler_path = {path}")
print(f"[setup] 样本 {len(samples)} 条")

client = trio.ServiceClient()

# ---------------------------------------------------------------- 1) 真实 loss 字段
print("\n=== 1) 训练一步，看真实 loss 字段 ===")
tc = client.create_lora_training_client(base_model=report["base_model"], rank=report["lora_rank"])
tok = tc.get_tokenizer()
s = samples[0]
msgs = s["messages"]
prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
prompt_ids = tok.encode(prompt_text)
answer_ids = tok.encode(msgs[-1]["content"])
eos = getattr(tok, "eos_token_id", None)
answer_ids = answer_ids + ([eos] if eos is not None else [])
tokens = prompt_ids + answer_ids
keep = len(prompt_ids) - 1
datum = trio.Datum(
    model_input=trio.ModelInput.from_ints(tokens[:-1]),
    loss_fn_inputs={"target_tokens": tokens[1:],
                    "weights": [0.0] * keep + [1.0] * (len(tokens) - 1 - keep)},
)
out = tc.forward_backward(data=[datum]).result()
print("  metrics =", getattr(out, "metrics", None))
lfo = getattr(out, "loss_fn_outputs", None)
if lfo:
    try:
        print("  loss_fn_outputs[0] keys =", list(lfo[0].keys()))
        print("  loss_fn_outputs[0] =", {k: (v if not hasattr(v, "__len__") or len(str(v)) < 60 else f"<{type(v).__name__}>") for k, v in lfo[0].items()})
    except Exception as exc:                                  # noqa: BLE001
        print("  loss_fn_outputs 读取失败:", exc)

# ---------------------------------------------------------------- 2) 用保存的权重推理
print("\n=== 2) 用 Sampler 权重推理（留出 5 条） ===")
sc = client.create_sampling_client(base_model=report["base_model"], model_path=path)
stok = sc.get_tokenizer()
params = trio.SamplingParams(max_tokens=64, seed=0, temperature=0.1)

hits = 0
for i, s in enumerate(samples[-5:]):
    msgs = s["messages"]
    prompt = stok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    ids = stok.encode(prompt)
    resp = sc.sample(prompt=trio.ModelInput.from_ints(ids), num_samples=1,
                     sampling_params=params).result()
    text = resp.sequences[0].text
    want = json.loads(msgs[-1]["content"])
    got = None
    try:
        blk = text[text.index("{"): text.rindex("}") + 1]
        got = json.loads(blk)
    except Exception:                                         # noqa: BLE001
        got = None
    ok = bool(got and got.get("action") == want["action"])
    hits += ok
    print(f"  [{i+1}] 期望={want['action']:14s} 得到={str(got.get('action') if got else None):14s} "
          f"{'✓' if ok else '✗'}  原文={text[:70]!r}")
print(f"\n命中率 {hits}/5")
