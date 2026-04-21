import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

from s3_routing_core import run_simulation


ALGORITHMS = [
    ("optimized", "Optimized"),
    ("hypatia", "Hypatia"),
    ("lsr", "LSR"),
    ("madrl", "MA-DRL"),
    ("otcp", "OTCP"),
    ("ftrl", "FTRL"),
    ("dtn_cgr", "DTN-CGR"),
]


def run_benchmark(sat_dir, uav_file, max_steps, save_outputs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    metrics_list = []

    for router_name, display_name in ALGORITHMS:
        print(f"[Benchmark] Running {display_name}...")
        metrics = run_simulation(
            router_name=router_name,
            sat_dir=sat_dir,
            uav_file=uav_file,
            save_outputs=save_outputs,
            max_steps=max_steps,
        )
        metrics_list.append(metrics)

    dataframe = pd.DataFrame(metrics_list)
    csv_path = os.path.join(output_dir, "algorithm_benchmark.csv")
    json_path = os.path.join(output_dir, "algorithm_benchmark.json")
    plot_path = os.path.join(output_dir, "algorithm_benchmark.png")

    dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(metrics_list, handle, ensure_ascii=False, indent=2)

    plot_benchmark(dataframe, plot_path)
    return dataframe, csv_path, json_path, plot_path


def plot_benchmark(dataframe, plot_path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    algorithms = dataframe["Algorithm"]
    chart_specs = [
        ("Runtime (s)", "Total Runtime (s)"),
        ("Route Success Rate (%)", "Route Success Rate (%)"),
        ("Avg Path Delay (ms)", "Average Path Delay (ms)"),
        ("Routing Compute Time (ms)", "Routing Compute Time (ms)"),
    ]
    colors = ["#355070", "#6d597a", "#b56576", "#e56b6f", "#2a9d8f", "#eaac8b", "#99c1b9"]

    for axis, (column_name, title) in zip(axes.flat, chart_specs):
        bars = axis.bar(algorithms, dataframe[column_name], color=colors[: len(dataframe)], width=0.65)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=20)
        for bar in bars:
            value = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + max(value * 0.02, 0.01),
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.suptitle("Seven Routing Algorithms Benchmark", fontsize=16)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Benchmark seven routing algorithms on the S3 project.")
    parser.add_argument("sat_dir", nargs="?", default="traces/sat_trace")
    parser.add_argument("--uav-file", dest="uav_file", default="traces/uav_trace_full.csv")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--save-outputs", action="store_true")
    parser.add_argument("--output-dir", default="outputs/benchmark_results")
    args = parser.parse_args()

    dataframe, csv_path, json_path, plot_path = run_benchmark(
        sat_dir=args.sat_dir,
        uav_file=args.uav_file,
        max_steps=args.max_steps,
        save_outputs=args.save_outputs,
        output_dir=args.output_dir,
    )

    print("\n=== Benchmark Summary ===")
    print(dataframe.to_string(index=False))
    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")
    print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()