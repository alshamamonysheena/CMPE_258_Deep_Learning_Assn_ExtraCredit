#!/usr/bin/env python3
import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd


def load_results(input_dir: str) -> pd.DataFrame:
    paths = [
        os.path.join(input_dir, "pytorch_benchmark.csv"),
        os.path.join(input_dir, "cublas_benchmark.csv"),
    ]
    frames = []
    for p in paths:
        if os.path.exists(p):
            frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError(
            f"No benchmark CSV files found under {input_dir}. "
            "Expected pytorch_benchmark.csv and/or cublas_benchmark.csv."
        )
    df = pd.concat(frames, ignore_index=True)
    df["case"] = (
        "B" + df["batch"].astype(str)
        + "-I" + df["in_features"].astype(str)
        + "-O" + df["out_features"].astype(str)
    )
    return df


def add_series(ax, df: pd.DataFrame, y_col: str, title: str, y_label: str) -> None:
    keys = sorted(df[["framework", "mode"]].drop_duplicates().itertuples(index=False, name=None))
    x_labels = sorted(df["case"].unique())
    x_pos = list(range(len(x_labels)))
    idx_map = {label: i for i, label in enumerate(x_labels)}

    for framework, mode in keys:
        part = df[(df["framework"] == framework) & (df["mode"] == mode)].copy()
        part["x"] = part["case"].map(idx_map)
        part = part.sort_values("x")
        label = f"{framework}:{mode}"
        ax.plot(part["x"], part[y_col], marker="o", linewidth=2, label=label)

    ax.set_title(title)
    ax.set_xlabel("FC size case")
    ax.set_ylabel(y_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.grid(alpha=0.3)
    ax.legend()


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot FC benchmark comparisons.")
    parser.add_argument("--input_dir", default="results", help="Directory containing benchmark CSV files")
    parser.add_argument("--output_dir", default="results", help="Directory to store output plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        df = load_results(args.input_dir)
    except Exception as e:
        print(f"ERROR: {e}")
        return 2

    fig, ax = plt.subplots(figsize=(12, 6))
    add_series(
        ax,
        df,
        y_col="avg_latency_ms",
        title="Average Forward Latency per FC Case",
        y_label="Latency (ms)",
    )
    latency_path = os.path.join(args.output_dir, "latency_comparison.png")
    fig.tight_layout()
    fig.savefig(latency_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    add_series(
        ax,
        df,
        y_col="throughput_tflops",
        title="Throughput Comparison per FC Case",
        y_label="Throughput (TFLOP/s)",
    )
    throughput_path = os.path.join(args.output_dir, "throughput_comparison.png")
    fig.tight_layout()
    fig.savefig(throughput_path, dpi=150)
    plt.close(fig)

    print(f"Saved: {latency_path}")
    print(f"Saved: {throughput_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
