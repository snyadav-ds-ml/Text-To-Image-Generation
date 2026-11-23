import csv
import os


def log_metrics(run_name, metrics, log_dir="results/metrics", merge=False):
    """Write metrics to CSV. If merge=True, merges with existing row instead of appending."""
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, f"{run_name}.csv")

    if merge and os.path.exists(csv_path):
        # read existing row and merge new metrics into it
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            existing = rows[-1]              # get last row
            existing.update(metrics)         # merge new values in
            metrics = existing

    with open(csv_path, "w", newline="") as f:   # always overwrite when merging
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)
