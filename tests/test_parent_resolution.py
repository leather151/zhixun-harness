"""验证 /previous 父检查点解析 + 交付目录（2026-11-15 平台试运行后修正后的规则）。

规则变了：**镜像自带模型优先**，`/previous`（平台父模型）只在镜像里没有该任务模型时兜底。
理由是平台每轮记录写着「retained the platform parent; no candidate was trained」——
它把自己的父模型挂 /previous，若让它优先，闭环就会从那个更弱的父模型起步，
没有挑战者上线时又把它原样交回去，白白丢掉镜像里已优化好的模型。
"""
import shutil
import sys
import tempfile
from pathlib import Path

APP = Path(r"D:\pycharm\competition")
sys.path.insert(0, str(APP / "harness"))

import runner  # noqa: E402
import tasks as T  # noqa: E402


class A:
    checkpoint = None
    previous = None
    model_out = None


tmp = Path(tempfile.mkdtemp(prefix="prev-"))
print("tmp =", tmp)

# 1) 什么都没挂 → 用内置交付模型
path, origin = runner.resolve_start_checkpoint("C", A())
print("无 /previous：", path, origin)
assert origin == "bundled" and path == T.checkpoint_path("C")

# 2) /previous 挂着平台父模型 → **仍然用内置交付模型**，但在 origin 里标出来
shutil.copy2(T.MODEL_ROOT / "C" / "checkpoint_official_baseline.npz", tmp / "checkpoint.npz")
a = A(); a.previous = str(tmp)
path, origin = runner.resolve_start_checkpoint("C", a)
print("有 /previous：", path, origin)
assert path == T.checkpoint_path("C"), "平台父模型不应覆盖镜像自带交付模型"
assert "platform parent" in origin

# 3) /previous 是单文件（也不该改变选择）
a = A(); a.previous = str(tmp / "checkpoint.npz")
path, origin = runner.resolve_start_checkpoint("C", a)
print("单文件 /previous：", path, origin)
assert path == T.checkpoint_path("C")

# 4) 镜像里没有该任务模型时，/previous 才兜底
import runner as R
orig = R.T.checkpoint_path
R.T.checkpoint_path = lambda task, name=None: tmp / "nope" / "checkpoint.npz"
try:
    (tmp / "sub").mkdir()
    shutil.copy2(APP / "models" / "C" / "checkpoint.npz", tmp / "sub" / "model.npz")
    a = A(); a.previous = str(tmp)
    path, origin = R.resolve_start_checkpoint("C", a)
    print("兜底 /previous：", path, origin)
    assert origin == "previous" and path.is_file()
finally:
    R.T.checkpoint_path = orig

# 5) --checkpoint 优先级最高
a = A(); a.checkpoint = str(APP / "models" / "C" / "checkpoint.npz"); a.previous = str(tmp)
print("显式 --checkpoint：", runner.resolve_start_checkpoint("C", a))
assert runner.resolve_start_checkpoint("C", a)[1] == "cli"

# 6) model_out_root 可写性探测
mo = tmp / "model-out"
a = A(); a.model_out = str(mo)
assert runner.model_out_root(a) == mo
print("model_out_root：", runner.model_out_root(a))

print("全部通过")
shutil.rmtree(tmp, ignore_errors=True)
