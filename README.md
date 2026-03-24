# CMPE 258 Extra Credit Report
## Performance Comparison: CUDA-Core FP32 GEMM vs Tensor Core TF32 GEMM

## 1. Objective
This project measures and explains the performance difference between a traditional CUDA-core GEMM path and a modern Tensor Core GEMM path for a simple fully connected (FC) forward pass.

Compared modes:
- **Mode 1 (Baseline):** FP32 matmul with TF32 disabled
- **Mode 2 (Tensor Core):** FP32 input/output with TF32 compute enabled

The benchmark is implemented in:
- PyTorch
- CUDA C++ with cuBLAS

## 2. Implementations

### 2.1 PyTorch benchmark
File: `benchmark_pytorch.py`

- FC forward pass implemented with `torch.nn.functional.linear`
- Baseline mode:
  - `torch.backends.cuda.matmul.allow_tf32 = False`
  - `torch.backends.cudnn.allow_tf32 = False`
- Tensor Core mode:
  - `torch.backends.cuda.matmul.allow_tf32 = True`
  - `torch.backends.cudnn.allow_tf32 = True`
- Timing method:
  - Warm-up iterations before timing
  - GPU timing via `torch.cuda.Event`
  - Explicit `torch.cuda.synchronize()`
- Reported metrics:
  - Average forward latency (ms)
  - Throughput (TFLOP/s)

### 2.2 CUDA C++ cuBLAS benchmark
File: `benchmark_cublas.cu`

- Baseline path: `cublasSgemm` (FP32)
- Tensor Core path: `cublasGemmEx`
  - input type: FP32
  - output type: FP32
  - compute type: `CUBLAS_COMPUTE_32F_FAST_TF32`
- Timing method:
  - Warm-up iterations
  - CUDA event timing
  - Device synchronization
- Reported metrics:
  - Average forward latency (ms)
  - Throughput (TFLOP/s)

## 3. Benchmark Methodology

### 3.1 FC size sweep
The same size sweep is used for all modes and both frameworks:

- `(B=256,  I=1024, O=1024)`
- `(B=512,  I=1024, O=2048)`
- `(B=512,  I=2048, O=2048)`
- `(B=1024, I=2048, O=2048)`
- `(B=1024, I=4096, O=4096)`

Where:
- `B` = batch size (`M`)
- `I` = input features (`K`)
- `O` = output features (`N`)

### 3.2 FLOPs and throughput
For FC forward GEMM:

\[
\text{FLOPs} = 2 \times M \times N \times K
\]

\[
\text{Throughput (TFLOP/s)} = \frac{\text{FLOPs}}{\text{latency\_seconds} \times 10^{12}}
\]

### 3.3 Timing protocol
- Warm-up iterations are run first to stabilize kernels/caches.
- Multiple timed iterations are averaged.
- GPU-side timing is used (not CPU wall-clock).

## 4. Reproducibility and Execution

### 4.1 Requirements
- NVIDIA GPU with CUDA support (Ampere+ recommended for TF32 Tensor Core gains)
- CUDA Toolkit (`nvcc` available in PATH)
- Python 3.9+
- PyTorch with CUDA support

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

### 4.2 Run all steps

```bash
python3 run_project.py
```

`run_project.py` performs:
1. PyTorch benchmark
2. cuBLAS benchmark compilation and execution
3. Plot generation
4. Exit with `sys.exit(exit_code)` semantics (`0` success, non-zero failure)

## 5. Output Artifacts
Generated under `results/`:

- `pytorch_benchmark.csv`
- `cublas_benchmark.csv`
- `latency_comparison.png`
- `throughput_comparison.png`

These artifacts are intended to be included in the submission repository.

## 6. Result Interpretation (What to discuss)
When writing your final observations, discuss:

- latency difference between baseline FP32 and TF32 Tensor Core mode
- throughput gain (TFLOP/s) for Tensor Core mode
- how speedup changes as matrix size increases
- why larger GEMMs usually show larger Tensor Core utilization
- any differences between PyTorch trends and direct cuBLAS trends

## 7. Deliverables Checklist
- [x] PyTorch implementation
- [x] CUDA C++ cuBLAS implementation
- [x] Baseline + Tensor Core benchmark modes
- [x] Warm-up and GPU timing protocol
- [x] Average latency and throughput metrics
- [x] Comparison plots
- [x] Self-verifiable exit status via `sys.exit(exit_code)`

## 8. Repository File Map
- `benchmark_pytorch.py` - PyTorch benchmark
- `benchmark_cublas.cu` - cuBLAS benchmark
- `plot_results.py` - plotting utility
- `run_project.py` - end-to-end runner
- `requirements.txt` - Python dependencies
- `results/` - generated CSVs and plots

## 9. Conclusion
This project provides a complete and reproducible comparison of CUDA-core FP32 GEMM and Tensor Core TF32 GEMM for FC forward-pass workloads in both high-level (PyTorch) and low-level (cuBLAS) implementations. The setup is designed for direct submission as a GitHub repository link.
# CMPE 258 Extra Credit: CUDA Core vs Tensor Core FC Benchmark

## 1) Objective

Measure and explain the performance difference between:

- **Mode 1 (CUDA Core baseline):** FP32 matmul with TF32 disabled
- **Mode 2 (Tensor Core path):** FP32 input/output with TF32 compute enabled

using a simple fully connected (FC) forward pass benchmark.

This project provides both:

- a **PyTorch implementation**
- a **CUDA C++ cuBLAS implementation**

with identical benchmark methodology and comparable matrix-size sweep.

## 2) What is implemented

### A. PyTorch benchmark (`benchmark_pytorch.py`)

- FC forward benchmark using `torch.nn.functional.linear`
- Mode 1: TF32 disabled (`allow_tf32=False`)
- Mode 2: TF32 enabled (`allow_tf32=True`)
- GPU timing using `torch.cuda.Event` and `torch.cuda.synchronize()`
- Warm-up iterations before timed iterations
- Reports average latency per forward pass (ms) and throughput (TFLOP/s)

### B. CUDA C++ benchmark (`benchmark_cublas.cu`)

- Baseline path: `cublasSgemm` (FP32)
- Tensor Core path: `cublasGemmEx` with:
  - FP32 input/output
  - `CUBLAS_COMPUTE_32F_FAST_TF32` compute mode
- Uses CUDA event timing, warm-up, and multiple timed iterations
- Writes benchmark metrics to CSV

### C. Plotting + orchestration

- `plot_results.py`: Generates comparison plots
  - `latency_comparison.png`
  - `throughput_comparison.png`
- `run_project.py`: Runs the full pipeline and exits with proper status code (`sys.exit`)

## 3) Benchmark methodology

- Same FC shape sweep is used in both implementations.
- For each case:
  1. run warm-up iterations
  2. run timed iterations on GPU
  3. compute average latency per forward pass
  4. compute throughput:

\[
\text{FLOPs} = 2 \times M \times N \times K,\quad
\text{TFLOP/s} = \frac{\text{FLOPs}}{\text{latency\_seconds} \times 10^{12}}
\]

## 4) Environment requirements

- NVIDIA GPU with CUDA support (Ampere+ recommended for clear TF32 Tensor Core gains)
- CUDA Toolkit (`nvcc`)
- Python 3.9+
- PyTorch with CUDA enabled

Install Python dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## 5) How to run (submission reproducibility)

Run end-to-end:

```bash
python3 run_project.py
```

Expected behavior:

1. Runs PyTorch benchmark
2. Compiles and runs CUDA C++ cuBLAS benchmark
3. Generates plots from CSV results
4. Returns exit code `0` on success, non-zero on failure

## 6) Output artifacts

Generated under `results/`:

- `pytorch_benchmark.csv`
- `cublas_benchmark.csv`
- `latency_comparison.png`
- `throughput_comparison.png`

These files are intended to be included in the final submission/repo.

## 7) Manual commands (optional)

Run only PyTorch benchmark:

```bash
python3 benchmark_pytorch.py --output results/pytorch_benchmark.csv
```

Compile cuBLAS benchmark:

```bash
nvcc -O3 -std=c++17 benchmark_cublas.cu -o benchmark_cublas -lcublas
```

Run cuBLAS benchmark:

```bash
./benchmark_cublas --output results/cublas_benchmark.csv
```

Generate plots:

```bash
python3 plot_results.py --input_dir results --output_dir results
```

## 8) Deliverables checklist (for this extra credit)

- [x] PyTorch implementation
- [x] CUDA C++ cuBLAS implementation
- [x] Two benchmark modes (baseline + Tensor Core TF32 path)
- [x] Warm-up + GPU timing + averaged latency
- [x] Throughput computation
- [x] Clear comparison plots
- [x] Self-verifiable run with `sys.exit(exit_code)` behavior
- [ ] Fill in observed results/discussion in report (use `report_template.md`)

## 9) Short result discussion guidance

In your write-up, discuss:

- latency and throughput difference between baseline and TF32 Tensor Core modes
- how speedup changes with matrix size
- why larger GEMMs typically show larger Tensor Core benefit
- any differences between PyTorch and raw cuBLAS measurements

You can use `report_template.md` for the final report text.

