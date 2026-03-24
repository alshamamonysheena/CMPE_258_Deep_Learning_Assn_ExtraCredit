#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys


def run_cmd(cmd, step_name):
    print(f"\n=== {step_name} ===")
    print(" ".join(cmd))
    proc = subprocess.run(cmd)
    print(f"[{step_name}] exit_code={proc.returncode}")
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run full FC benchmark project end-to-end.")
    parser.add_argument("--results_dir", default="results", help="Directory for CSVs and plots")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations")
    parser.add_argument(
        "--skip_cublas",
        action="store_true",
        help="Skip CUDA C++ cuBLAS benchmark (not recommended for final submission)",
    )
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    py_csv = os.path.join(args.results_dir, "pytorch_benchmark.csv")
    cb_csv = os.path.join(args.results_dir, "cublas_benchmark.csv")

    # 1) PyTorch benchmark
    code = run_cmd(
        [
            sys.executable,
            "benchmark_pytorch.py",
            "--output",
            py_csv,
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
        ],
        "PyTorch benchmark",
    )
    if code != 0:
        return code

    # 2) CUDA C++ benchmark
    if not args.skip_cublas:
        nvcc = shutil.which("nvcc")
        if not nvcc:
            print("ERROR: nvcc not found in PATH, but cuBLAS benchmark is required.")
            return 3

        code = run_cmd(
            [
                nvcc,
                "-O3",
                "-std=c++17",
                "benchmark_cublas.cu",
                "-o",
                "benchmark_cublas",
                "-lcublas",
            ],
            "Compile cuBLAS benchmark",
        )
        if code != 0:
            return code

        code = run_cmd(
            [
                "./benchmark_cublas",
                "--output",
                cb_csv,
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
            ],
            "Run cuBLAS benchmark",
        )
        if code != 0:
            return code

    # 3) Plot results
    code = run_cmd(
        [
            sys.executable,
            "plot_results.py",
            "--input_dir",
            args.results_dir,
            "--output_dir",
            args.results_dir,
        ],
        "Generate plots",
    )
    if code != 0:
        return code

    print("\nAll required steps completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
