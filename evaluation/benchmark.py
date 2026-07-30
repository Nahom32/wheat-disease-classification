import os
import json
import pandas as pd


BENCHMARK_DIR = "outputs/benchmark"
BENCHMARK_CSV = os.path.join(BENCHMARK_DIR, "experiment_comparison.csv")
BENCHMARK_JSON = os.path.join(BENCHMARK_DIR, "experiment_comparison.json")


def format_value(v):
    if isinstance(v, float):
        return round(v, 4)
    return v


def write_benchmark(results: list, output_dir: str = None):
    """Write a CSV and JSON summary comparing all experiments.

    Args:
        results: list of summary dicts from pipeline.run_experiment().
        output_dir: override output directory (default: outputs/benchmark/).
    """
    out_dir = output_dir or BENCHMARK_DIR
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for r in results:
        row = {
            "experiment": r["experiment"],
            "test_f1_micro": format_value(r["test_f1_micro"]),
            "test_f1_macro": format_value(r["test_f1_macro"]),
        }
        for cls_name, f1_val in r.get("per_class_f1", {}).items():
            row[f"f1_{cls_name}"] = format_value(f1_val)
        rows.append(row)

    df = pd.DataFrame(rows)

    csv_path = os.path.join(out_dir, "experiment_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"Benchmark CSV saved to {csv_path}")

    json_path = os.path.join(out_dir, "experiment_comparison.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Benchmark JSON saved to {json_path}")

    return df
