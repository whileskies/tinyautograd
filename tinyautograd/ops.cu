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

__global__ void add_vec_kernel(const float *a, const float *b, float *c, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

extern "C" void add_vec(const float *a, const float *b, float *c, int n) {
    int block = 1024;
    int grid = (n + block - 1) / block;
    add_vec_kernel<<<grid, block>>>(a, b, c, n);
    cudaDeviceSynchronize();
}


__global__ void add_vec_broadcast_kernel(const float *a, const float *b, float *c,
                int n, int b_stride, int broadcast_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int b_idx;
    if (broadcast_dim == 0) {  // 广播 b = (1, D)
        b_idx = idx % b_stride; // 重复每行
    } else if (broadcast_dim == 1) {
        b_idx = idx / b_stride;
    } else {
        b_idx = 0;
    }

    // printf("a: %.2f, b: %.2f\n", a[idx], b[b_idx]);
    c[idx] = a[idx] + b[b_idx];
}

extern "C" void add_vec_broadcast(
    const float *a, const float *b, float *c,
    int total_size, int b_stride, int b_dim)
{
    int block = 1024;
    int grid = (total_size + block - 1) / block;
    add_vec_broadcast_kernel<<<grid, block>>>(
        a, b, c, total_size, b_stride, b_dim
    );
    cudaDeviceSynchronize();
}