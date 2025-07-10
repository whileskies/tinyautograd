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


__global__ void power_local_grad_kernel(const float* x, float* grad, float p, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        grad[idx] = p * powf(x[idx], p - 1.0f);
    }
}

extern "C" void launch_power_grad(const float* x, float* grad, float p, int n) {
    int block = 1024;
    int grid = (n + block - 1) / block;
    power_local_grad_kernel<<<grid, block>>>(x, grad, p, n);
    cudaDeviceSynchronize();
}


// reduce
__global__ void reduce_sum_kernel(const float *input, float *output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

extern "C" void launch_sum(float *d_input, float *d_output, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    float *d_temp = NULL;
    cudaMalloc(&d_temp, blocks * sizeof(float));

    reduce_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_temp, n);

    int s = blocks;
    while (s > 1) {
        int threads2 = (s > 256) ? 256 : s;
        int blocks2 = (s + threads2 - 1) / threads2;
        reduce_sum_kernel<<<blocks2, threads2, threads2 * sizeof(float)>>>(d_temp, d_temp, s);
    }

    cudaMemcpy(d_output, d_temp, sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(d_temp);
}

__global__ void sum_axis0_kernel(const float *input, float *output, int N, int M) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= M) return;

    float sum = 0;
    for (int row = 0; row < N; row++) {
        sum += input[row * M + col];
    }

    output[col] = sum;
}

extern "C" void launch_sum_axis0(const float *d_input, float *d_output, int N, int M) {
    int blockSize = 256;
    int gridSize = (M + blockSize - 1) / blockSize;
    sum_axis0_kernel<<<gridSize, blockSize>>>(d_input, d_output, N, M);
}


// __global__ void sum_axis1_kernel(const float *input, float *output, int N, int M) {
//     int row = blockIdx.x * blockDim.x + threadIdx.x;
//     if (row >= N) return;

//     float sum = 0;
//     for (int col = 0; col < M; col++) {
//         sum += input[row * M + col];
//     }

//     output[row] = sum;
// }

// extern "C" void launch_sum_axis1(const float *d_input, float *d_output, int N, int M) {
//     int blockSize = 256;
//     int gridSize = (N + blockSize - 1) / blockSize;
//     sum_axis0_kernel<<<gridSize, blockSize>>>(d_input, d_output, N, M);
// }

__global__ void reduce_axis1_stage1(const float* input, float* partial, int N, int M, int stride) {
    int row = blockIdx.y;
    int seg = blockIdx.x;  // segment/block index along the row
    int tid = threadIdx.x;

    int start = seg * stride;
    int end = min(start + stride, M);

    extern __shared__ float sdata[];
    float sum = 0.0f;

    for (int i = start + tid; i < end; i += blockDim.x) {
        sum += input[row * M + i];
    }

    sdata[tid] = sum;
    __syncthreads();

    // intra-block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    // store local result
    if (tid == 0)
        partial[row * gridDim.x + seg] = sdata[0];
}

__global__ void reduce_axis1_stage2(const float* partial, float* output, int N, int num_partials) {
    int row = blockIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < num_partials; ++i) {
        sum += partial[row * num_partials + i];
    }
    output[row] = sum;
}


extern "C" void launch_sum_axis1(const float* d_input, float* d_output, int N, int M) {
    int stride = 1024;
    int num_seg = (M + stride - 1) / stride;
    int threads = 256;

    float* d_partial = nullptr;
    cudaMalloc(&d_partial, sizeof(float) * N * num_seg);

    dim3 grid(num_seg, N);
    reduce_axis1_stage1<<<grid, threads, threads * sizeof(float)>>>(d_input, d_partial, N, M, stride);

    reduce_axis1_stage2<<<N, 1>>>(d_partial, d_output, N, num_seg);
    cudaFree(d_partial);
}