#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " -> " << cudaGetErrorString(err__) << std::endl;    \
            return 1;                                                         \
        }                                                                     \
    } while (0)

#define CHECK_CUBLAS(call)                                                    \
    do {                                                                      \
        cublasStatus_t st__ = (call);                                         \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                  \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__    \
                      << " -> status=" << static_cast<int>(st__) << std::endl;\
            return 1;                                                         \
        }                                                                     \
    } while (0)

struct Case {
    int m;
    int k;
    int n;
};

static std::vector<Case> get_sweep() {
    return {
        {256, 1024, 1024},
        {512, 1024, 2048},
        {512, 2048, 2048},
        {1024, 2048, 2048},
        {1024, 4096, 4096},
    };
}

struct Config {
    std::string output = "results/cublas_benchmark.csv";
    int warmup = 30;
    int iters = 100;
};

static Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--output" && i + 1 < argc) {
            cfg.output = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            cfg.warmup = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            cfg.iters = std::stoi(argv[++i]);
        }
    }
    return cfg;
}

static double elapsed_ms_for_events(cudaEvent_t start, cudaEvent_t end) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, end);
    return static_cast<double>(ms);
}

static int benchmark_sgemm(
    cublasHandle_t handle, const Case& c, int warmup, int iters, double& avg_ms_out) {
    const int m = c.m;
    const int k = c.k;
    const int n = c.n;

    float *A = nullptr, *B = nullptr, *C = nullptr;
    CHECK_CUDA(cudaMalloc(&A, sizeof(float) * m * k));
    CHECK_CUDA(cudaMalloc(&B, sizeof(float) * k * n));
    CHECK_CUDA(cudaMalloc(&C, sizeof(float) * m * n));

    CHECK_CUDA(cudaMemset(A, 0, sizeof(float) * m * k));
    CHECK_CUDA(cudaMemset(B, 0, sizeof(float) * k * n));
    CHECK_CUDA(cudaMemset(C, 0, sizeof(float) * m * n));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    for (int i = 0; i < warmup; ++i) {
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m,
            n,
            k,
            &alpha,
            A,
            m,
            B,
            k,
            &beta,
            C,
            m));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CHECK_CUBLAS(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m,
            n,
            k,
            &alpha,
            A,
            m,
            B,
            k,
            &beta,
            C,
            m));
    }
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));

    double total_ms = elapsed_ms_for_events(start, end);
    avg_ms_out = total_ms / static_cast<double>(iters);

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}

static int benchmark_gemmex_tf32(
    cublasHandle_t handle, const Case& c, int warmup, int iters, double& avg_ms_out) {
    const int m = c.m;
    const int k = c.k;
    const int n = c.n;

    float *A = nullptr, *B = nullptr, *C = nullptr;
    CHECK_CUDA(cudaMalloc(&A, sizeof(float) * m * k));
    CHECK_CUDA(cudaMalloc(&B, sizeof(float) * k * n));
    CHECK_CUDA(cudaMalloc(&C, sizeof(float) * m * n));

    CHECK_CUDA(cudaMemset(A, 0, sizeof(float) * m * k));
    CHECK_CUDA(cudaMemset(B, 0, sizeof(float) * k * n));
    CHECK_CUDA(cudaMemset(C, 0, sizeof(float) * m * n));

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));

    for (int i = 0; i < warmup; ++i) {
        CHECK_CUBLAS(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m,
            n,
            k,
            &alpha,
            A,
            CUDA_R_32F,
            m,
            B,
            CUDA_R_32F,
            k,
            &beta,
            C,
            CUDA_R_32F,
            m,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, end;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&end));

    CHECK_CUDA(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        CHECK_CUBLAS(cublasGemmEx(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            m,
            n,
            k,
            &alpha,
            A,
            CUDA_R_32F,
            m,
            B,
            CUDA_R_32F,
            k,
            &beta,
            C,
            CUDA_R_32F,
            m,
            CUBLAS_COMPUTE_32F_FAST_TF32,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }
    CHECK_CUDA(cudaEventRecord(end));
    CHECK_CUDA(cudaEventSynchronize(end));

    double total_ms = elapsed_ms_for_events(start, end);
    avg_ms_out = total_ms / static_cast<double>(iters);

    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);

    if (!std::filesystem::path(cfg.output).parent_path().empty()) {
        std::filesystem::create_directories(std::filesystem::path(cfg.output).parent_path());
    }

    int dev_count = 0;
    cudaError_t dev_err = cudaGetDeviceCount(&dev_count);
    if (dev_err != cudaSuccess || dev_count <= 0) {
        std::cerr << "ERROR: No CUDA device available." << std::endl;
        return 2;
    }

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    std::ofstream ofs(cfg.output);
    if (!ofs.is_open()) {
        std::cerr << "ERROR: Failed to open output CSV: " << cfg.output << std::endl;
        cublasDestroy(handle);
        return 3;
    }

    ofs << "framework,mode,batch,in_features,out_features,avg_latency_ms,throughput_tflops,warmup_iters,timed_iters,timestamp_s\n";

    auto sweep = get_sweep();
    for (const auto& c : sweep) {
        double avg_ms = 0.0;
        if (benchmark_sgemm(handle, c, cfg.warmup, cfg.iters, avg_ms) != 0) {
            cublasDestroy(handle);
            return 4;
        }
        const double flops = 2.0 * static_cast<double>(c.m) * c.n * c.k;
        const double tflops = flops / (avg_ms * 1e-3) / 1e12;
        const auto now_s = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        std::cout << "[cuBLAS][cuda_core_fp32_sgemm] "
                  << "B=" << c.m << ", I=" << c.k << ", O=" << c.n
                  << " | latency=" << std::fixed << std::setprecision(4) << avg_ms
                  << " ms | throughput=" << std::setprecision(3) << tflops << " TFLOP/s\n";

        ofs << "cublas,cuda_core_fp32_sgemm,"
            << c.m << "," << c.k << "," << c.n << ","
            << std::fixed << std::setprecision(6) << avg_ms << ","
            << std::fixed << std::setprecision(6) << tflops << ","
            << cfg.warmup << "," << cfg.iters << ","
            << std::fixed << std::setprecision(3) << now_s << "\n";
    }

    for (const auto& c : sweep) {
        double avg_ms = 0.0;
        if (benchmark_gemmex_tf32(handle, c, cfg.warmup, cfg.iters, avg_ms) != 0) {
            cublasDestroy(handle);
            return 5;
        }
        const double flops = 2.0 * static_cast<double>(c.m) * c.n * c.k;
        const double tflops = flops / (avg_ms * 1e-3) / 1e12;
        const auto now_s = std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        std::cout << "[cuBLAS][tensor_core_tf32_gemmex] "
                  << "B=" << c.m << ", I=" << c.k << ", O=" << c.n
                  << " | latency=" << std::fixed << std::setprecision(4) << avg_ms
                  << " ms | throughput=" << std::setprecision(3) << tflops << " TFLOP/s\n";

        ofs << "cublas,tensor_core_tf32_gemmex,"
            << c.m << "," << c.k << "," << c.n << ","
            << std::fixed << std::setprecision(6) << avg_ms << ","
            << std::fixed << std::setprecision(6) << tflops << ","
            << cfg.warmup << "," << cfg.iters << ","
            << std::fixed << std::setprecision(3) << now_s << "\n";
    }

    ofs.close();
    CHECK_CUBLAS(cublasDestroy(handle));
    std::cout << "Saved cuBLAS results to: " << cfg.output << std::endl;
    return 0;
}
