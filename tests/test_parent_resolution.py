"""验证 /previous 父检查点解析与 /model 交付目录（不跑完整闭环，先打两个函数）。"""
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
print("无 /previous：", runner.resolve_start_checkpoint("C", A()))
assert runner.resolve_start_checkpoint("C", A())[1] == "bundled"

# 2) /previous 是目录，里面有 checkpoint.npz
shutil.copy2(T.checkpoint_path("C"), tmp / "checkpoint.npz")
a = A(); a.previous = str(tmp)
path, origin = runner.resolve_start_checkpoint("C", a)
print("目录布局：", path, origin)
assert origin == "previous" and path.name == "checkpoint.npz"

# 3) /previous 是单文件
a = A(); a.previous = str(tmp / "checkpoint.npz")
path, origin = runner.resolve_start_checkpoint("C", a)
print("单文件布局：", path, origin)
assert origin == "previous"

# 4) 平台自定义布局 <TASK>/checkpoint.npz
(tmp / "C").mkdir()
shutil.move(str(tmp / "checkpoint.npz"), str(tmp / "C" / "checkpoint.npz"))
a = A(); a.previous = str(tmp)
path, origin = runner.resolve_start_checkpoint("C", a)
print("子目录布局：", path, origin)
assert origin == "previous" and path.parent.name == "C"

# 5) 随机子目录（兜底 rglob）
(tmp / "weird" / "nested").mkdir(parents=True)
shutil.move(str(tmp / "C" / "checkpoint.npz"), str(tmp / "weird" / "nested" / "model.npz"))
a = A(); a.previous = str(tmp)
path, origin = runner.resolve_start_checkpoint("C", a)
print("兜底扫描：", path, origin)
assert origin == "previous" and path.name == "model.npz"

# 6) --checkpoint 优先级最高
a = A(); a.checkpoint = str(APP / "models" / "C" / "checkpoint_best.npz"); a.previous = str(tmp)
print("显式 --checkpoint：", runner.resolve_start_checkpoint("C", a))
assert runner.resolve_start_checkpoint("C", a)[1] == "cli"

# 7) model_out_root 可写性探测
mo = tmp / "model-out"
a = A(); a.model_out = str(mo)
print("model_out_root：", runner.model_out_root(a))
assert runner.model_out_root(a) == mo

print("全部通过")
shutil.rmtree(tmp, ignore_errors=True)
