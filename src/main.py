from pathlib import Path
import hashlib
import time
import zipfile

import joblib
import numpy as np
import pandas as pd
import swanlab
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


# main.py 位于 src 目录，因此上一级是 competition 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "gas_drift.zip"
RESULT_DIR = BASE_DIR / "results"

N_FEATURES = 128


def file_sha256(path: Path) -> str:
    """计算文件 SHA256，用于实验记录和数据版本追溯。"""
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def read_batch(archive, filename):
    """读取一个气体传感器批次。"""
    text = archive.read(filename).decode("latin1")

    features = []
    labels = []

    for line in text.splitlines():
        tokens = line.split()
        if not tokens:
            continue

        # 第一列是类别，原始类别编号为 1 到 6
        label = int(float(tokens[0])) - 1

        # 后面是 1:value 2:value ... 格式的 128 维特征
        row = np.zeros(N_FEATURES, dtype=np.float64)
        for token in tokens[1:]:
            index_text, value_text = token.split(":", 1)
            index = int(index_text) - 1
            row[index] = float(value_text)

        features.append(row)
        labels.append(label)

    return np.asarray(features), np.asarray(labels)


def calculate_psi(reference, current, bins=10):
    """计算单个特征的 Population Stability Index。"""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))

    if len(edges) < 3:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    reference_counts, _ = np.histogram(reference, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    reference_share = reference_counts / max(reference.size, 1)
    current_share = current_counts / max(current.size, 1)

    reference_share = np.clip(reference_share, 1e-6, None)
    current_share = np.clip(current_share, 1e-6, None)

    psi = np.sum(
        (current_share - reference_share)
        * np.log(current_share / reference_share)
    )
    return float(psi)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到数据文件：{DATA_PATH}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    data_sha256 = file_sha256(DATA_PATH)

    run = swanlab.init(
        project="zhixun-engine-3tasks",
        experiment_name="A-gas-rf-baseline",
        config={
            "task": "A",
            "task_type": "gas_sensor_classification",
            "model": "RandomForestClassifier",
            "n_estimators": 200,
            "class_weight": "balanced",
            "random_seed": 42,
            "train_batch": 1,
            "stream_batches": "2-10",
            "n_features": N_FEATURES,
            "dataset_sha256": data_sha256,
            "split_note": "开发用切分；正式评测以官方SDK为准",
            "drift_rule": "mean_psi>=0.10 or max_psi>=0.25",
        },
    )

    try:
        with zipfile.ZipFile(DATA_PATH) as archive:
            file_map = {
                Path(name).name: name
                for name in archive.namelist()
                if name.endswith(".dat")
            }

            train_x, train_y = read_batch(
                archive,
                file_map["batch1.dat"],
            )

            print("训练数据形状：", train_x.shape)
            print("训练类别数量：", len(np.unique(train_y)))

            model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced",
            )

            training_started = time.perf_counter()
            model.fit(train_x, train_y)
            training_seconds = time.perf_counter() - training_started

            results = []

            # batch1 用于训练，batch2-batch10 模拟按时间到来的数据流
            for batch_id in range(2, 11):
                filename = file_map[f"batch{batch_id}.dat"]
                batch_x, batch_y = read_batch(archive, filename)

                predictions = model.predict(batch_x)

                accuracy = float(
                    accuracy_score(batch_y, predictions)
                )
                macro_f1 = float(
                    f1_score(
                        batch_y,
                        predictions,
                        average="macro",
                        zero_division=0,
                    )
                )

                psi_values = [
                    calculate_psi(
                        train_x[:, feature_index],
                        batch_x[:, feature_index],
                    )
                    for feature_index in range(N_FEATURES)
                ]

                mean_psi = float(np.mean(psi_values))
                max_psi = float(np.max(psi_values))
                drift_alert = int(
                    mean_psi >= 0.10 or max_psi >= 0.25
                )

                swanlab.log(
                    {
                        "batch": batch_id,
                        "rows": len(batch_y),
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                        "mean_psi": mean_psi,
                        "max_psi": max_psi,
                        "drift_alert": drift_alert,
                    },
                    step=batch_id,
                )

                results.append(
                    {
                        "batch": batch_id,
                        "rows": len(batch_y),
                        "accuracy": accuracy,
                        "macro_f1": macro_f1,
                        "mean_psi": mean_psi,
                        "max_psi": max_psi,
                        "drift_alert": drift_alert,
                    }
                )

                print(
                    f"batch {batch_id:02d}: "
                    f"accuracy={accuracy:.4f}, "
                    f"macro_f1={macro_f1:.4f}, "
                    f"mean_psi={mean_psi:.4f}, "
                    f"max_psi={max_psi:.4f}"
                )

        metrics = pd.DataFrame(results)
        metrics.to_csv(
            RESULT_DIR / "metrics_by_batch.csv",
            index=False,
            encoding="utf-8-sig",
        )

        joblib.dump(
            model,
            RESULT_DIR / "gas_baseline.joblib",
        )

        swanlab.log(
            {
                "training_seconds": training_seconds,
                "train_rows": len(train_y),
                "best_accuracy": float(metrics["accuracy"].max()),
                "best_macro_f1": float(metrics["macro_f1"].max()),
                "worst_accuracy": float(metrics["accuracy"].min()),
                "worst_macro_f1": float(metrics["macro_f1"].min()),
                "alert_batches": int(metrics["drift_alert"].sum()),
            }
        )

        print("\n逐批次结果：")
        print(metrics.round(4).to_string(index=False))
        print(f"\n模型训练耗时：{training_seconds:.3f} 秒")
        print(f"结果目录：{RESULT_DIR}")
        print(f"SwanLab Run：{run}")

    finally:
        swanlab.finish()


if __name__ == "__main__":
    main()