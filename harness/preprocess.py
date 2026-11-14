"""特征预处理：秩变换（ECDF / 分位映射）。

为什么用秩变换（B 包的结论，见 03_实验/数据分析/B_分析.md）：
- 34 维特征量级与分布差异极大，且轴承之间有系统性水平偏移；
- 把每个特征映射成"在训练集总体里的分位"后，绝对水平被抹平，同一退化阶段变得可比；
- 对单调漂移天然鲁棒（只依赖序关系，不依赖均值方差）。

合规：
- 分位表**只在官方 train.jsonl 上拟合**，validation 只做 apply；
- 分位表随模型一起保存（写进 checkpoint 的 metadata），推理时按同一张表变换，保证一致。

实测（B 包岭回归探针）：原始特征 RMSE 501 → 秩特征 529（尺度错）→ 秩 + 按轴承等权 **293**。
所以秩变换必须和"按组等权"一起用。
"""
from __future__ import annotations

import numpy as np

N_QUANTILES = 201
DIGITS = 7          # 写入 metadata 时保留的小数位，控制 checkpoint 体积


def fit_rank_table(x, n=N_QUANTILES):
    """在训练集上拟合分位表，返回 (d, n) 数组。"""
    x = np.asarray(x, dtype=np.float64)
    grid = np.linspace(0.0, 1.0, n)
    table = np.empty((x.shape[1], n), dtype=np.float64)
    for j in range(x.shape[1]):
        col = np.quantile(x[:, j], grid)
        # 让每列严格递增，np.interp 才稳定（重复分位点加极小步长）
        table[j] = np.maximum.accumulate(col + np.arange(n) * 1e-12)
    return table


def apply_rank(table, x):
    """把 x 映射到训练集分位（0–1）；超出训练范围的值自动夹到 0 或 1。"""
    table = np.asarray(table, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    grid = np.linspace(0.0, 1.0, table.shape[1])
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = np.interp(x[:, j], table[j], grid)
    return out


def table_to_meta(table):
    """分位表 → 可写进 checkpoint metadata 的紧凑格式。"""
    return {"kind": "rank", "n_quantiles": int(table.shape[1]),
            "quantiles": [[round(float(v), DIGITS) for v in row] for row in table]}


def table_from_meta(meta):
    """从 checkpoint metadata 取回分位表；没有就返回 None（= 原始特征模型）。"""
    if not isinstance(meta, dict):
        return None
    pre = meta.get("preprocess")
    if not pre or pre.get("kind") != "rank" or "quantiles" not in pre:
        return None
    return np.asarray(pre["quantiles"], dtype=np.float64)


def transform_inputs(model_meta, x):
    """按模型自带的分位表变换输入；模型没有预处理就是恒等。"""
    table = table_from_meta(model_meta)
    return np.asarray(x, dtype=np.float64) if table is None else apply_rank(table, x)


def transform_jsonl_line(line, table):
    """把一行 JSONL 的数值输入就地替换为秩特征（用于喂给官方 train_run 续训）。"""
    import json
    row = json.loads(line)
    paths, values = _flatten(row["inputs"])
    if table is None:
        return line
    rank = apply_rank(table, np.asarray([values], dtype=np.float64))[0]
    row["inputs"] = _rebuild(paths, rank)
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


def _flatten(inputs):
    import json
    paths, values = [], []

    def visit(value, path):
        if isinstance(value, dict):
            for key in sorted(value):
                visit(value[key], path + [("key", key)])
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [("index", index)])
        else:
            paths.append(json.loads(json.dumps(path)))
            values.append(float(value))

    visit(inputs, [])
    return paths, values


def _rebuild(paths, values):
    """把扁平数值按路径还原成嵌套 dict/list（顺序与官方 flatten 一致：键按字母序）。"""
    out = {}
    for path, value in zip(paths, values):
        cursor = out
        for i, (kind, key) in enumerate(path):
            last = i == len(path) - 1
            if last:
                if kind == "key":
                    cursor[key] = float(value)
                else:
                    cursor[key] = float(value)
            else:
                nkind, nkey = path[i + 1]
                if kind == "key":
                    if key not in cursor:
                        cursor[key] = [] if nkind == "index" else {}
                    cursor = cursor[key]
                else:
                    while len(cursor) <= key:
                        cursor.append(None)
                    if cursor[key] is None:
                        cursor[key] = [] if nkind == "index" else {}
                    cursor = cursor[key]
    return out
