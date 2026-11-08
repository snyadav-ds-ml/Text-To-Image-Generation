import csv  # write metrics to file
import os   # create directories


def log_metrics(run_name, metrics, log_dir="results/metrics"):
    os.makedirs(log_dir, exist_ok=True)              # create folder if missing

    csv_path = os.path.join(log_dir, f"{run_name}.csv")

    write_header = not os.path.exists(csv_path)      # first write needs header

    with open(csv_path, "a", newline="") as f:        # append, not overwrite
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if write_header:
            writer.writeheader()                      # column names on first row
        writer.writerow(metrics)                      # one row per run
