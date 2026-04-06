import csv
import os
from datetime import datetime
from pathlib import Path


def log_experiment(params, metrics, top_features):
    file_path = "experiments_log.csv"
    headers = params.keys() | metrics.keys() | {"timestamp", "top_feature"}

    file_exists = os.path.isfile(file_path)
    file_path = Path(__file__).parent / file_path
    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                **params,
                **metrics,
                "top_feature": top_features[0],
            }
        )
