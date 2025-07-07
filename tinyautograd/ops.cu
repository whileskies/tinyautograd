#include <cuda_runtime.h>
#include <iostream>

extern "C" float* move_to_gpu(float *a, int n) {
    float *d_a;
    cudaMalloc((void **)&d_a, n * sizeof(float));
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    return d_a;
}

extern "C" void move_to_cpu(float* a, float* d_a, int n) {
    cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
}

extern "C" float* alloc_on_gpu(int n) {
    float *d_ptr;
    cudaMalloc((void**)&d_ptr, n * sizeof(float));
    return d_ptr;
}

extern "C" void free_gpu_memory(float *d_ptr) {
    cudaFree(d_ptr);
}

struct AddOp {
    __device__ float operator()(float x, float y) const {
        return x + y;
    }
};

struct MulOp {
    __device__ float operator()(float x, float y) const {
        return x * y;
    }
};

template <typename OP>
__global__ void elementwise_kernel(const float *a, const float *b, float *c, int n, OP op) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = op(a[idx], b[idx]);
    }
}

template <typename OP>
__global__ void elementwise_broadcast_kernel(const float *a, const float *b, float *c, int n, int dim, int bcast_type, OP op) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n) return;

    float bv = 0;
    if (bcast_type == 0)       bv = b[idx % dim];         // row
    else if (bcast_type == 1)  bv = b[idx / dim];         // col
    else if (bcast_type == -1) bv = b[0];                 // scalar
    c[idx] = op(a[idx], bv);
}


extern "C" void add_vec(const float *a, const float *b, float *c, int n) {
    int block = 1024;
    int grid = (n + block - 1) / block;
    elementwise_kernel<<<grid, block>>>(a, b, c, n, AddOp());
    cudaDeviceSynchronize();
}


extern "C" void add_vec_broadcast(
    const float *a, const float *b, float *c,
    int total_size, int b_stride, int b_dim)
{
    int block = 1024;
    int grid = (total_size + block - 1) / block;
    elementwise_broadcast_kernel<<<grid, block>>>(
        a, b, c, total_size, b_stride, b_dim, AddOp()
    );
    cudaDeviceSynchronize();
}

extern "C" void mul_vec(const float *a, const float *b, float *c, int n) {
    int block = 1024;
    int grid = (n + block - 1) / block;
    elementwise_kernel<<<grid, block>>>(a, b, c, n, MulOp());
    cudaDeviceSynchronize();
}


extern "C" void mul_vec_broadcast(
    const float *a, const float *b, float *c,
    int total_size, int b_stride, int b_dim)
{
    int block = 1024;
    int grid = (total_size + block - 1) / block;
    elementwise_broadcast_kernel<<<grid, block>>>(
        a, b, c, total_size, b_stride, b_dim, MulOp()
    );
    cudaDeviceSynchronize();
}


template <typename UnaryOp>
__global__ void unary_elementwise_kernel(const float *x, float *y, int n, UnaryOp op) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        y[idx] = op(x[idx]);
    }
}

struct ReluOp {
    __device__ float operator()(float x) const { return x > 0.0f ? x : 0.0f; }
};

struct ReluGradOp {
    __device__ float operator()(float x) const {
        return x > 0 ? 1.0f : 0.0f;
    }
};

struct LogOp {
    __device__ float operator()(float x) const { return logf(x); }
};

struct LogGradOp {
    __device__ float operator()(float x) const { 
        return 1.0f / x;
     }
};

struct TanhOp {
    __device__ float operator()(float x) const { return tanhf(x); }
};

struct TanhGradOp {
    __device__ float operator()(float x) const {
        float t = tanhf(x);
        return 1.0f - t * t;
    }
};

extern "C" void launch_relu(const float *x, float *y, int n) {
    int block = 1024;
    int grid = (n + block - 1) / block;
    unary_elementwise_kernel<<<grid, block>>>(x, y, n, ReluOp());
}

extern "C" void launch_relu_grad(const float *x, float *y, int n) {
    int block = 1024;
    int grid = (n + block - 1) / block;
    unary_elementwise_kernel<<<grid, block>>>(x, y, n, ReluGradOp());
}

extern "C" void launch_log(const float *x, float *y, int n) {
    int block = 1024;
    int grid = (n + block - 1) / block;
    unary_elementwise_kernel<<<grid, block>>>(x, y, n, LogOp());
}

extern "C" void launch_log_grad(const float *x, float *y, int n) {
    int block = 1024;
    int grid = (n + block - 1) / block;
    unary_elementwise_kernel<<<grid, block>>>(x, y, n, LogGradOp());
}

extern "C" void launch_tanh(const float *x, float *y, int n) {
    int block = 1024;
    int grid = (n + block - 1) / block;
    unary_elementwise_kernel<<<grid, block>>>(x, y, n, TanhOp());
}

extern "C" void launch_tanh_grad(const float *x, float *y, int n) {
    int block = 1024;
    int grid = (n + block - 1) / block;
    unary_elementwise_kernel<<<grid, block>>>(x, y, n, TanhGradOp());
}


__global__ void power_kernel(const float* x, float* y, float p, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        y[idx] = powf(x[idx], p);
    }
}

extern "C" void launch_power(const float* x, float* y, float p, int n) {
    int block = 1024;
    int grid = (n + block - 1) / block;
    power_kernel<<<grid, block>>>(x, y, p, n);
    cudaDeviceSynchronize();
}

