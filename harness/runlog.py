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
    def __init__(self, outdir, swan="auto", experiment="harness", config=None,
                 project=None, mode=None):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.windows_path = self.outdir / "windows.jsonl"
        self.decisions_path = self.outdir / "decisions.jsonl"
        self.summary_path = self.outdir / "summary.json"
        self._windows = self.windows_path.open("a", encoding="utf-8")
        self._decisions = self.decisions_path.open("a", encoding="utf-8")
        self.swan = None
        self.swan_error = None
        self.swan_url = None
        self.swan_project = project or os.environ.get("SWANLAB_PROJECT") or "zhixun-engine-3tasks"
        self.swan_experiment = experiment
        if swan == "off" or (swan == "auto" and os.environ.get("SWANLAB_MODE") == "disabled"):
            return
        wanted = mode or os.environ.get("SWANLAB_MODE") or "cloud"
        for attempt in (wanted, "local"):
            try:
                import swanlab
                self.swan = swanlab.init(project=self.swan_project, experiment_name=experiment,
                                        config=config or {}, mode=attempt)
                self.swan_url = _swan_url(self.swan)
                self.swan_mode = attempt
                break
            except Exception as exc:                              # noqa: BLE001
                self.swan_error = f"{type(exc).__name__}: {exc}"
                self.swan = None

    # ---------------------------------------------------------------- 写
    def log_window(self, payload, metrics=None):
        self._write(self._windows, payload)
        if self.swan is None:
            return
        step = payload.get("window_index", 0)
        values = metrics if metrics is not None else {
            k: v for k, v in payload.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}
        values = {k: v for k, v in values.items()
                  if isinstance(v, (int, float)) and not isinstance(v, bool)}
        values["window"] = step
        if values:
            try:
                self.swan.log(values, step=step)
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


def _swan_url(run):
    """尽量取出云端运行页 URL（不同 swanlab 版本字段不一样）。"""
    for attr in ("url", "run_url", "cloud_url"):
        value = getattr(run, attr, None)
        if isinstance(value, str) and value.startswith("http"):
            return value
    try:
        settings = getattr(run, "settings", None) or getattr(run, "_settings", None)
        for attr in ("url", "cloud_url"):
            value = getattr(settings, attr, None)
            if isinstance(value, str) and value.startswith("http"):
                return value
    except Exception:                                             # noqa: BLE001
        pass
    return None


def open_swan(project=None, experiment="harness", config=None, mode=None):
    """独立于 RunLogger 的 SwanLab 入口（给 train_b.py 这类实验脚本用）。

    返回 dict(run, url, project, experiment, mode, error)。
    云端失败会退到 local，两者都失败则 run=None 并把原因带回来（不抛异常，不阻塞实验）。
    """
    proj = project or os.environ.get("SWANLAB_PROJECT") or "zhixun-engine-3tasks"
    wanted = mode or os.environ.get("SWANLAB_MODE") or "cloud"
    last_error = None
    for attempt in (wanted, "local"):
        try:
            import swanlab
            run = swanlab.init(project=proj, experiment_name=experiment,
                               config=config or {}, mode=attempt)
            return {"run": run, "url": _swan_url(run), "project": proj,
                    "experiment": experiment, "mode": attempt, "error": None}
        except Exception as exc:                                  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
    return {"run": None, "url": None, "project": proj, "experiment": experiment,
            "mode": None, "error": last_error}
