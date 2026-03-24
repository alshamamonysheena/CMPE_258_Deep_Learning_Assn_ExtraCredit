# CMPE 258 Extra Credit Report
## CUDA-Core FP32 GEMM vs Tensor Core TF32 GEMM (FC Forward Benchmark)

## 1. Objective
This project measures and explains the performance difference between:

- **Mode 1 (baseline):** FP32 matmul with TF32 disabled
- **Mode 2 (Tensor Core path):** FP32 input/output with TF32 compute enabled

for a simple fully connected (FC) forward pass.

The benchmark is implemented in:
- PyTorch
- CUDA C++ with cuBLAS

## 2. Implementation Summary

### 2.1 PyTorch benchmark
File: `benchmark_pytorch.py`

- FC forward pass via `torch.nn.functional.linear`
- Baseline mode: TF32 disabled
- Tensor Core mode: TF32 enabled
- Warm-up before timing
- GPU timing with `torch.cuda.Event` and `torch.cuda.synchronize()`
- Reports average latency (ms) and throughput (TFLOP/s)

### 2.2 CUDA C++ cuBLAS benchmark
File: `benchmark_cublas.cu`

- Baseline path: `cublasSgemm` (FP32)
- Tensor Core path: `cublasGemmEx` with `CUBLAS_COMPUTE_32F_FAST_TF32`
- FP32 input/output in both paths
- Warm-up + CUDA event timing
- Reports average latency (ms) and throughput (TFLOP/s)

### 2.3 Orchestration and plotting

- `run_project.py`: end-to-end run and exit status via `sys.exit(exit_code)`
- `plot_results.py`: generates:
  - `results/latency_comparison.png`
  - `results/throughput_comparison.png`

## 3. Benchmark Methodology

### 3.1 FC size sweep
Same sweep for all modes and both frameworks:

- `(B=256,  I=1024, O=1024)`
- `(B=512,  I=1024, O=2048)`
- `(B=512,  I=2048, O=2048)`
- `(B=1024, I=2048, O=2048)`
- `(B=1024, I=4096, O=4096)`

Where:
- `B` = batch size (`M`)
- `I` = input features (`K`)
- `O` = output features (`N`)

### 3.2 Throughput formula

\[
\text{FLOPs} = 2 \times M \times N \times K
\]

\[
\text{Throughput (TFLOP/s)} = \frac{\text{FLOPs}}{\text{latency\_seconds} \times 10^{12}}
\]

## 4. Experimental Environment (Colab Run)

- Platform: Google Colab Pro
- GPU: `NVIDIA A100-SXM4-40GB`
- Driver CUDA version (from `nvidia-smi`): `13.0`
- `nvcc` version: `12.8`
- Warm-up iterations: `30`
- Timed iterations: `100`

## 5. Results

### 5.1 PyTorch results

| FC Case (B, I, O) | Baseline FP32 no TF32 (ms) | Tensor Core TF32 (ms) | Speedup (x) | Baseline Throughput (TFLOP/s) | TF32 Throughput (TFLOP/s) |
|---|---:|---:|---:|---:|---:|
| 256, 1024, 1024 | 0.0603 | 0.0215 | 2.80 | 8.904 | 24.990 |
| 512, 1024, 2048 | 0.1785 | 0.0364 | 4.90 | 12.031 | 59.008 |
| 512, 2048, 2048 | 0.3210 | 0.0582 | 5.52 | 13.380 | 73.740 |
| 1024, 2048, 2048 | 0.5774 | 0.1078 | 5.36 | 14.877 | 79.687 |
| 1024, 4096, 4096 | 1.9380 | 0.3267 | 5.93 | 17.730 | 105.163 |

### 5.2 cuBLAS results

| FC Case (B, I, O) | Baseline `cublasSgemm` (ms) | TF32 `cublasGemmEx` (ms) | Speedup (x) | Baseline Throughput (TFLOP/s) | TF32 Throughput (TFLOP/s) |
|---|---:|---:|---:|---:|---:|
| 256, 1024, 1024 | 0.0591 | 0.0296 | 2.00 | 9.079 | 18.110 |
| 512, 1024, 2048 | 0.1577 | 0.0321 | 4.91 | 13.621 | 67.002 |
| 512, 2048, 2048 | 0.2983 | 0.0579 | 5.15 | 14.398 | 74.222 |
| 1024, 2048, 2048 | 0.5832 | 0.0941 | 6.20 | 14.729 | 91.270 |
| 1024, 4096, 4096 | 2.0970 | 0.3277 | 6.40 | 16.385 | 104.844 |

### 5.3 Key observations

- TF32 Tensor Core path is faster than baseline FP32 in all benchmarked FC sizes.
- Speedup increases with matrix size, reaching about `5.9x` in PyTorch and `6.4x` in cuBLAS at the largest case.
- Throughput scales from roughly `~9-18 TFLOP/s` baseline to `~75-105 TFLOP/s` with TF32 Tensor Core path on larger cases.

## 6. Discussion

The measured behavior is consistent with expected Tensor Core acceleration on Ampere GPUs. The baseline mode enforces standard FP32 math without TF32 acceleration, while the Tensor Core mode allows TF32 compute for FP32 GEMM operations. As problem size increases, fixed overheads are amortized and matrix multiply kernels better saturate Tensor Core compute resources, which explains the larger gains on bigger GEMMs.

PyTorch and direct cuBLAS results show the same trend and similar absolute performance at larger sizes. Minor differences are expected due to framework-level dispatch and kernel selection details.

## 7. Reproducibility

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run full pipeline:

```bash
python3 run_project.py
```

Expected outputs in `results/`:

- `pytorch_benchmark.csv`
- `cublas_benchmark.csv`
- `latency_comparison.png`
- `throughput_comparison.png`

## 8. Submission Checklist

- [x] PyTorch implementation
- [x] CUDA C++ cuBLAS implementation
- [x] Baseline FP32 and Tensor Core TF32 benchmark modes
- [x] Warm-up and GPU timing protocol
- [x] Average latency and throughput reporting
- [x] Comparison plots
- [x] Self-verifiable exit status (`sys.exit(exit_code)`)
- [x] Results included in repository (`results/`)
