"""运行日志：本地 JSONL + 可选 SwanLab。

SwanLab 是**可选**的：
- 本地开发/比赛记录用它（`--swan on|auto`）；
- 封闭评测环境没有网络，用 `--swan off` 或环境变量 `SWANLAB_MODE=disabled`，
  本模块会静默跳过，绝不让日志组件成为"跑不完"的原因。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


class RunLogger:
    def __init__(self, outdir, swan="auto", experiment="harness", config=None):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.windows_path = self.outdir / "windows.jsonl"
        self.decisions_path = self.outdir / "decisions.jsonl"
        self.summary_path = self.outdir / "summary.json"
        self._windows = self.windows_path.open("a", encoding="utf-8")
        self._decisions = self.decisions_path.open("a", encoding="utf-8")
        self.swan = None
        self.swan_error = None
        want = (swan == "on") or (swan == "auto")
        if want and os.environ.get("SWANLAB_MODE", "local") != "disabled":
            try:
                import swanlab
                self.swan = swanlab.init(project="zhixun-engine", experiment_name=experiment,
                                        config=config or {},
                                        mode=os.environ.get("SWANLAB_MODE", "local"))
            except Exception as exc:                              # noqa: BLE001
                self.swan_error = f"{type(exc).__name__}: {exc}"

    # ---------------------------------------------------------------- 写
    def log_window(self, payload):
        self._write(self._windows, payload)
        if self.swan is not None:
            try:
                flat = {f"w_{k}": v for k, v in payload.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool)}
                flat["window"] = payload.get("window_index", 0)
                self.swan.log(flat, step=payload.get("window_index", 0))
            except Exception:                                     # noqa: BLE001
                pass

    def log_decision(self, payload):
        self._write(self._decisions, payload)

    def summary(self, payload):
        self.summary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=_default) + "\n",
            encoding="utf-8")
        if self.swan is not None:
            try:
                self.swan.log({k: v for k, v in payload.items()
                               if isinstance(v, (int, float)) and not isinstance(v, bool)})
            except Exception:                                     # noqa: BLE001
                pass

    def close(self):
        for handle in (self._windows, self._decisions):
            try:
                handle.close()
            except Exception:                                     # noqa: BLE001
                pass
        if self.swan is not None:
            try:
                self.swan.finish()
            except Exception:                                     # noqa: BLE001
                pass

    @staticmethod
    def _write(handle, payload):
        handle.write(json.dumps(payload, ensure_ascii=False, default=_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _default(obj):
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:                                             # noqa: BLE001
        pass
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
