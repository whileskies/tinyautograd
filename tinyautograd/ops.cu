#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <cfloat>

#define BLOCK_SZ 256

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

extern "C" void copy_to_cpu(float* a, float* d_a, int n) {
    cudaMemcpy(a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);
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
    int block = BLOCK_SZ;
    int grid = (n + block - 1) / block;
    elementwise_kernel<<<grid, block>>>(a, b, c, n, AddOp());
    // cudaDeviceSynchronize();
}


extern "C" void add_vec_broadcast(
    const float *a, const float *b, float *c,
    int total_size, int b_stride, int b_dim)
{
    int block = BLOCK_SZ;
    int grid = (total_size + block - 1) / block;
    elementwise_broadcast_kernel<<<grid, block>>>(
        a, b, c, total_size, b_stride, b_dim, AddOp()
    );
    // cudaDeviceSynchronize();
}

extern "C" void mul_vec(const float *a, const float *b, float *c, int n) {
    int block = BLOCK_SZ;
    int grid = (n + block - 1) / block;
    elementwise_kernel<<<grid, block>>>(a, b, c, n, MulOp());
    // cudaDeviceSynchronize();
}


extern "C" void mul_vec_broadcast(
    const float *a, const float *b, float *c,
    int total_size, int b_stride, int b_dim)
{
    int block = BLOCK_SZ;
    int grid = (total_size + block - 1) / block;
    elementwise_broadcast_kernel<<<grid, block>>>(
        a, b, c, total_size, b_stride, b_dim, MulOp()
    );
    // cudaDeviceSynchronize();
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
        return 1.0f - x * x;  // x = tanhf(x) already
    }
};

struct SigmoidOp {
    __device__ float operator()(float x) const {
        return 1.0f / (1.0f + expf(-x));
    }
};

struct SigmoidGradOp {
    __device__ float operator()(float x) const {
        return x * (1.0f - x);  // x = sigmoid(x) already
    }
};

struct ExpOp {
    __device__ float operator()(float x) const {
        return expf(x);  
    }
};

extern "C" void launch_relu(const float *x, float *y, int n) {
    int block = BLOCK_SZ;
    int grid = (n + block - 1) / block;
    unary_elementwise_kernel<<<grid, block>>>(x, y, n, ReluOp());
}

extern "C" void launch_relu_grad(const float *x, float *y, int n) {
    int block = BLOCK_SZ;
    int grid = (n + block - 1) / block;
    unary_elementwise_kernel<<<grid, block>>>(x, y, n, ReluGradOp());
}

extern "C" void launch_log(const float *x, float *y, int n) {
    int block = BLOCK_SZ;
    int grid = (n + block - 1) / block;
    unary_elementwise_kernel<<<grid, block>>>(x, y, n, LogOp());
}

extern "C" void launch_log_grad(const float *x, float *y, int n) {
    int block = BLOCK_SZ;
    int grid = (n + block - 1) / block;
    unary_elementwise_kernel<<<grid, block>>>(x, y, n, LogGradOp());
}

extern "C" void launch_tanh(const float *x, float *y, int n) {
    int block = BLOCK_SZ;
    int grid = (n + block - 1) / block;
    unary_elementwise_kernel<<<grid, block>>>(x, y, n, TanhOp());
}

extern "C" void launch_tanh_grad(const float *x, float *y, int n) {
    int block = BLOCK_SZ;
    int grid = (n + block - 1) / block;
    unary_elementwise_kernel<<<grid, block>>>(x, y, n, TanhGradOp());
}

extern "C" void launch_sigmoid(const float *x, float *y, int n) {
    int block = BLOCK_SZ;
    int grid = (n + block - 1) / block;
    unary_elementwise_kernel<<<grid, block>>>(x, y, n, SigmoidOp());
}

extern "C" void launch_sigmoid_grad(const float *x, float *y, int n) {
    int block = BLOCK_SZ;
    int grid = (n + block - 1) / block;
    unary_elementwise_kernel<<<grid, block>>>(x, y, n, SigmoidGradOp());
}

extern "C" void launch_exp(const float *x, float *y, int n) {
    int block = BLOCK_SZ;
    int grid = (n + block - 1) / block;
    unary_elementwise_kernel<<<grid, block>>>(x, y, n, ExpOp());
}

__global__ void power_kernel(const float* x, float* y, float p, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        y[idx] = powf(x[idx], p);
    }
}

extern "C" void power(const float* x, float* y, float p, int n) {
    int block = BLOCK_SZ;
    int grid = (n + block - 1) / block;
    power_kernel<<<grid, block>>>(x, y, p, n);
    // cudaDeviceSynchronize();
}


__global__ void power_local_grad_kernel(const float* x, float* grad, float p, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        grad[idx] = p * powf(x[idx], p - 1.0f);
    }
}

extern "C" void power_grad(const float* x, float* grad, float p, int n) {
    int block = BLOCK_SZ;
    int grid = (n + block - 1) / block;
    power_local_grad_kernel<<<grid, block>>>(x, grad, p, n);
    // cudaDeviceSynchronize();
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

extern "C" void sum(float *d_input, float *d_output, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    float *d_temp = NULL;
    cudaMalloc(&d_temp, blocks * sizeof(float));

    reduce_sum_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_input, d_temp, n);

    int s = blocks;
    while (s > 1) {
        int threads2 = 256;
        int blocks2 = (s + threads2 - 1) / threads2;
        reduce_sum_kernel<<<blocks2, threads2, threads2 * sizeof(float)>>>(d_temp, d_temp, s);
        s = blocks2;
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

extern "C" void sum_axis0(const float *d_input, float *d_output, int N, int M) {
    int blockSize = BLOCK_SZ;
    int gridSize = (M + blockSize - 1) / blockSize;
    sum_axis0_kernel<<<gridSize, blockSize>>>(d_input, d_output, N, M);
}


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


extern "C" void sum_axis1(const float* d_input, float* d_output, int N, int M) {
    int stride = BLOCK_SZ;
    int num_seg = (M + stride - 1) / stride;
    int threads = 256;

    float* d_partial = nullptr;
    cudaMalloc(&d_partial, sizeof(float) * N * num_seg);

    dim3 grid(num_seg, N);
    reduce_axis1_stage1<<<grid, threads, threads * sizeof(float)>>>(d_input, d_partial, N, M, stride);

    reduce_axis1_stage2<<<N, 1>>>(d_partial, d_output, N, num_seg);
    cudaFree(d_partial);
}


__global__ void fill_kernel(float *data, float value, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        data[i] = value;
    }
}


extern "C" void fill(float *d_data, float value, int n) {
    int blockSize = BLOCK_SZ;
    int gridSize = (n + blockSize - 1) / blockSize;
    fill_kernel<<<gridSize, blockSize>>>(d_data, value, n);
}

__global__ void print(float *a, float *b) {
    printf("%.2f %.2f %.2f %.2f\n", a[0], a[1], a[2], a[3]);
    printf("%.2f %.2f %.2f %.2f\n", b[0], b[1], b[2], b[3]);
}


extern "C" void matmul(float *mat_a, float *mat_b, float *mat_c, int M, int N, int K) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const float alpha = 1.0f, beta = 0.0f;

    // Correct parameters for C = A * B where A, B, C are row-major.
    // This is achieved by computing C^T = B^T * A^T using cublasSgemm (which is column-major).
    // So, d_mat_b (which holds B) acts as the first matrix (B^T)
    // and d_mat_a (which holds A) acts as the second matrix (A^T).
    cublasSgemm(
        handle,
        CUBLAS_OP_N, // First operand (d_mat_b, which is B) is treated as B^T (column-major). No transpose needed from cuBLAS.
        CUBLAS_OP_N, // Second operand (d_mat_a, which is A) is treated as A^T (column-major). No transpose needed from cuBLAS.
        N,           // Result matrix (C^T) rows: N (original C's columns)
        M,           // Result matrix (C^T) columns: M (original C's rows)
        K,           // Inner dimension: K
        &alpha,
        mat_b,       // Pointer to B (KxN row-major)
        N,           // Leading dimension of B: N (original column count of B)
        mat_a,       // Pointer to A (MxK row-major)
        K,           // Leading dimension of A: K (original column count of A)
        &beta,
        mat_c,       // Pointer to C (MxN row-major, will be filled as C^T in column-major)
        N);          // Leading dimension of C^T: N (original column count of C)

    cublasDestroy(handle);
}


// Tile 大小
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_shared(const float* __restrict__ input, float* __restrict__ output, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  
    // +1 是为了避免共享内存bank conflict

    int x = blockIdx.x * TILE_DIM + threadIdx.x;  // 列索引
    int y = blockIdx.y * TILE_DIM + threadIdx.y;  // 行索引

    // 先读入 tile 中，注意分多行读完tile（threadIdx.y < BLOCK_ROWS）
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < M && x < N) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * N + x];
        }
    }

    __syncthreads();

    // 计算转置后的位置
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // 转置后 x,y 交换
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // 写回输出矩阵，分多行写
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if ((y + j) < N && x < M) {
            output[(y + j) * M + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

extern "C" void transpose_matrix(float *mat_a, float *mat_c, int M, int N) {
    dim3 blockDim(TILE_DIM, BLOCK_ROWS);
    dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    transpose_shared<<<gridDim, blockDim>>>(mat_a, mat_c, M, N);

    // // cudaDeviceSynchronize();
}


__global__ void softmax_kernel_simple(const float *logits, float *probs, int B, int C) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= B) return;

    const float *row_logits = logits + row * C;
    float *row_probs = probs + row * C;

    // Step 1: 找最大值
    float max_val = -FLT_MAX;
    for (int i = tid; i < C; i += blockDim.x) {
        float val = row_logits[i];
        max_val = fmaxf(max_val, val);
    }

    // warp reduce max (简化，无 __shared__)
    for (int offset = 16; offset > 0; offset /= 2)
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));

    // Step 2: 计算 exp 和 sum
    float sum_exp = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        float e = __expf(row_logits[i] - max_val);
        row_probs[i] = e;
        sum_exp += e;
    }

    // warp reduce sum
    for (int offset = 16; offset > 0; offset /= 2)
        sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, offset);

    // Step 3: 归一化
    for (int i = tid; i < C; i += blockDim.x) {
        row_probs[i] = row_probs[i] / sum_exp;
    }
}

__global__ void cross_entropy_loss_kernel_simple(const float *probs, const float *target, float *loss, int B, int C) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= B) return;

    const float *row_probs = probs + row * C;
    const float *row_target = target + row * C;

    float local_loss = 0.0f;
    for (int i = tid; i < C; i += blockDim.x) {
        float t = row_target[i];
        float p = row_probs[i];
        local_loss += -t * logf(p + 1e-9f);
    }

    // warp reduce loss
    for (int offset = 16; offset > 0; offset /= 2)
        local_loss += __shfl_xor_sync(0xffffffff, local_loss, offset);

    if (tid == 0)
        loss[row] = local_loss;
}

extern "C" void launch_softmax_simple(const float *logits, float *probs, int B, int C) {
    dim3 grid(B);
    dim3 block(32);  // warp size
    softmax_kernel_simple<<<grid, block>>>(logits, probs, B, C);
}

extern "C" void launch_cross_entropy_loss_simple(const float *probs, const float *target, float *loss, int B, int C) {
    dim3 grid(B);
    dim3 block(32);
    cross_entropy_loss_kernel_simple<<<grid, block>>>(probs, target, loss, B, C);
}



__global__ void softmax_cross_entropy_grad(const float *probs, const float *target, float *grad, int B, int C) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int total = B * C;
    if (idx >= total) return;

    grad[idx] = probs[idx] - target[idx];
}

extern "C" void launch_softmax_cross_entropy_grad(const float *probs, const float *target, float *grad, int B, int C) {
    int total = B * C;
    int block = BLOCK_SZ;
    int grid = (total + block - 1) / block;
    softmax_cross_entropy_grad<<<grid, block>>>(probs, target, grad, B, C);
}

