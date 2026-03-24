#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import List

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Case:
    batch: int
    in_features: int
    out_features: int

    @property
    def m(self) -> int:
        return self.batch

    @property
    def k(self) -> int:
        return self.in_features

    @property
    def n(self) -> int:
        return self.out_features

    @property
    def flops(self) -> float:
        # GEMM FLOPs for forward: 2 * M * N * K
        return 2.0 * self.m * self.n * self.k


SWEEP: List[Case] = [
    Case(256, 1024, 1024),
    Case(512, 1024, 2048),
    Case(512, 2048, 2048),
    Case(1024, 2048, 2048),
    Case(1024, 4096, 4096),
]


def benchmark_case(case: Case, warmup: int, iters: int) -> float:
    x = torch.randn(case.batch, case.in_features, device="cuda", dtype=torch.float32)
    w = torch.randn(case.out_features, case.in_features, device="cuda", dtype=torch.float32)
    b = torch.randn(case.out_features, device="cuda", dtype=torch.float32)

    with torch.inference_mode():
        for _ in range(warmup):
            _ = F.linear(x, w, b)
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            _ = F.linear(x, w, b)
        end.record()
        torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    return elapsed_ms / iters


def run_mode(
    mode_name: str,
    allow_tf32: bool,
    warmup: int,
    iters: int,
    writer: csv.DictWriter,
) -> None:
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

    # Encourage high-performance kernels while still preserving FP32 IO.
    if allow_tf32:
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("highest")

    for case in SWEEP:
        avg_ms = benchmark_case(case, warmup, iters)
        tflops = case.flops / (avg_ms * 1e-3) / 1e12
        writer.writerow(
            {
                "framework": "pytorch",
                "mode": mode_name,
                "batch": case.batch,
                "in_features": case.in_features,
                "out_features": case.out_features,
                "avg_latency_ms": f"{avg_ms:.6f}",
                "throughput_tflops": f"{tflops:.6f}",
                "warmup_iters": warmup,
                "timed_iters": iters,
                "timestamp_s": f"{time.time():.3f}",
            }
        )
        print(
            f"[PyTorch][{mode_name}] B={case.batch}, I={case.in_features}, "
            f"O={case.out_features} | latency={avg_ms:.4f} ms | throughput={tflops:.3f} TFLOP/s"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="PyTorch FC benchmark: CUDA core vs Tensor Core TF32")
    parser.add_argument("--output", default="results/pytorch_benchmark.csv", help="CSV output path")
    parser.add_argument("--warmup", type=int, default=30, help="Warmup iterations per case")
    parser.add_argument("--iters", type=int, default=100, help="Timed iterations per case")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available in PyTorch. This benchmark requires a CUDA GPU.")
        return 2

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32

    try:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "framework",
                "mode",
                "batch",
                "in_features",
                "out_features",
                "avg_latency_ms",
                "throughput_tflops",
                "warmup_iters",
                "timed_iters",
                "timestamp_s",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Mode 1: FP32 matmul, TF32 disabled (CUDA-core baseline intent)
            run_mode("cuda_core_fp32_no_tf32", False, args.warmup, args.iters, writer)

            # Mode 2: FP32 input/output with TF32 compute enabled (Tensor Core path)
            run_mode("tensor_core_tf32", True, args.warmup, args.iters, writer)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32
        # Reset to default recommendation
        torch.set_float32_matmul_precision("high")

    print(f"Saved PyTorch results to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
