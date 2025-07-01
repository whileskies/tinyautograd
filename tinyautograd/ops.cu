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