"""用 PyTRiO 把策略层"决策模型"做一次 LoRA SFT，并保存为 **Sampler 权重**（拿到 trio 路径）。

数据：`data/sft/policy_sft.jsonl`（由 harness/build_policy_sft.py 从 57 次真实闭环轨迹里抽的
87 条「窗口状态 → 动作 + 理由」监督对，可回溯到 runs/*/windows.jsonl 与 SwanLab run）。

训练：base=`Qwen/Qwen3.5-4B`，LoRA rank 32，cross_entropy + `weights` 掩码（只对"回答"部分计损失），
小批量 + 少量 epoch（这是策略行为克隆，不是知识注入）。

产出：
1. `save_weights_for_sampler(name="zhixun-policy-v1")` 返回的 Sampler 权重路径（表单要填的就是它）；
2. `runs/trio_sft_report.json`：task_id / model_id / 步数 / 损失曲线 / 路径 / 复采样验证结果。

用法：
    python harness/trio_sft_policy.py --epochs 3 --rank 32
    python harness/trio_sft_policy.py --dry-run     # 只建训练客户端，不训练
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
SFT = APP / "data" / "sft" / "policy_sft.jsonl"
REPORT = APP / "runs" / "trio_sft_report.json"


def load_key():
    if os.environ.get("PYTRIO_API_KEY"):
        return os.environ["PYTRIO_API_KEY"]
    for name in ("PYTRIO_API_KEY", "SAMPLER_API_KEY"):
        env = APP / ".env.local"
        if env.is_file():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith(name + "="):
                    return line.split("=", 1)[1].strip()
    raise SystemExit("找不到 PYTRIO_API_KEY（环境变量或 .env.local）")


def build_batch(tok, samples, eos_id):
    """把一批 messages 转成 Datum：prompt 位置权重 0，只学回答部分。"""
    import pytrio as trio

    data = []
    for s in samples:
        msgs = s["messages"]
        prompt_text = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        answer_text = msgs[-1]["content"]
        prompt_ids = tok.encode(prompt_text)
        answer_ids = tok.encode(answer_text)
        if eos_id is not None:
            answer_ids = answer_ids + [eos_id]
        tokens = prompt_ids + answer_ids
        if len(tokens) < 3:
            continue
        input_ids = tokens[:-1]
        target_ids = tokens[1:]
        # 前 len(prompt_ids)-1 个位置是"读题"，不计损失
        keep = len(prompt_ids) - 1
        weights = [0.0] * keep + [1.0] * (len(target_ids) - keep)
        data.append(trio.Datum(
            model_input=trio.ModelInput.from_ints(input_ids),
            loss_fn_inputs={"target_tokens": target_ids, "weights": weights},
        ))
    return data


def main(argv=None):
    ap = argparse.ArgumentParser(description="PyTRiO 策略模型 LoRA SFT + 保存 Sampler 权重")
    ap.add_argument("--base-model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--name", default="zhixun-policy-v1")
    ap.add_argument("--dry-run", action="store_true", help="只创建训练客户端，不提交训练步")
    args = ap.parse_args(argv)

    os.environ["PYTRIO_API_KEY"] = load_key()
    import pytrio as trio

    samples = [json.loads(line) for line in SFT.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"[data] {len(samples)} 条 SFT 样本（{SFT}）")

    client = trio.ServiceClient()
    print(f"[svc ] 可用模型 {client.get_supported_models()}")
    t0 = time.time()
    tc = client.create_lora_training_client(base_model=args.base_model, rank=args.rank)
    print(f"[train] 训练客户端 OK：run={getattr(tc, 'training_run_cuid', '?')} model_id={tc.model_id} lora={tc.lora} ({time.time()-t0:.1f}s)")
    if args.dry_run:
        print("[dry-run] 不提交训练步。")
        return 0

    tok = tc.get_tokenizer()
    eos_id = getattr(tok, "eos_token_id", None)
    print(f"[tok  ] tokenizer OK（eos={eos_id}）")

    losses, steps = [], 0
    t1 = time.time()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(samples)))
        # 固定顺序即可（数据量小，shuffle 收益可忽略）；用 epoch 做偏移保证批次组合略有变化
        order = order[epoch % args.batch:] + order[:epoch % args.batch]
        epoch_loss = []
        for start in range(0, len(order), args.batch):
            batch = [samples[i] for i in order[start:start + args.batch]]
            data = build_batch(tok, batch, eos_id)
            if not data:
                continue
            out = tc.forward_backward(data=data).result()
            tc.optim_step(trio.AdamParams(learning_rate=args.learning_rate)).result()
            steps += 1
            m = getattr(out, "metrics", None) or {}
            loss = None
            # PyTRiO 实测返回的是 loss_sum / loss_mean / token_count，其余键名只作兼容
            for key in ("loss_mean", "loss", "mean_loss", "cross_entropy_loss", "loss:mean"):
                if isinstance(m, dict) and key in m:
                    loss = float(m[key])
                    break
            if loss is None:
                try:
                    el = out.loss_fn_outputs[0]["elementwise_loss"]
                    vals = getattr(el, "data", el)
                    loss = float(sum(vals) / max(1, len(vals)))
                except Exception:                             # noqa: BLE001
                    loss = float("nan")
            epoch_loss.append(loss)
            losses.append(loss)
            print(f"  epoch {epoch} step {steps:3d}  loss={loss:.4f}  ({time.time()-t1:.1f}s)")
        if epoch_loss:
            print(f"[epoch {epoch}] mean loss = {sum(epoch_loss)/len(epoch_loss):.4f}")

    print("[save ] save_weights_for_sampler ...")
    res = tc.save_weights_for_sampler(name=args.name).result()
    print("[save ] 返回：", res)
    path = getattr(res, "path", None) or (res.get("path") if isinstance(res, dict) else None)
    if not path:
        dump = getattr(res, "model_dump", None)
        path = (dump() or {}).get("path") if callable(dump) else None
    print("[save ] Sampler 权重路径 =", path)

    report = {
        "base_model": args.base_model,
        "lora_rank": args.rank,
        "epochs": args.epochs,
        "batch": args.batch,
        "learning_rate": args.learning_rate,
        "steps": steps,
        "n_samples": len(samples),
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_mean": round(sum(losses) / len(losses), 5) if losses else None,
        "loss_curve": [round(x, 5) for x in losses],
        "task_id": str(getattr(tc, "training_run_cuid", "")),
        "model_id": str(getattr(tc, "model_id", "")),
        "sampler_path": path,
        "checkpoint_name": args.name,
        "train_seconds": round(time.time() - t1, 1),
        "dataset": str(SFT.relative_to(APP)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[done ] 报告 -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

