from .rawtensor import RawTensor
import numpy as np
import time
import random

def assert_equal(res, res_g):
    assert np.allclose(res.data, res_g.data, rtol=1e-5, atol=1e-8)
    assert res.shape == res_g.shape

def print_res(cpu, gpu):
    print()
    print('-------cpu--------')
    print('shape:', cpu.shape, ", data:")
    print(cpu.data)
    print('-------gpu--------')
    print('shape:', gpu.shape, ", data:")
    print(gpu.data)
    print('------------------')

def test_add_vec():
    a = [[1, 1, 1], [1, 1, 1]]
    b = [[2, 2, 2], [2, 2, 2]]
    c = [[3, 3, 3], [3, 3, 3]]
    ta = RawTensor(a)
    tb = RawTensor(b)
    tc = RawTensor(c)
    res= ta + tb + tc
    # print(res.data)
    # print(res.shape)
    ta.to('cuda')
    tb.to('cuda')
    tc.to('cuda')

    res_g = ta + tb + tc
    res_g.to('cpu')
    # print(res_g.data, res_g.shape)
    assert_equal(res, res_g)
    print_res(res, res_g)

def test_sub_vec():
    a = [[1, 1, 1], [1, 1, 1]]
    b = [[2, 2, 2], [2, 2, 2]]
    c = [[3, 3, 3], [3, 3, 3]]
    ta = RawTensor(a)
    tb = RawTensor(b)
    tc = RawTensor(c)
    res= ta - tb - tc
    # print(res.data)
    # print(res.shape)
    ta.to('cuda')
    tb.to('cuda')
    tc.to('cuda')

    res_g = ta - tb - tc
    res_g.to('cpu')
    # print(res_g.data, res_g.shape)
    assert_equal(res, res_g)
    print_res(res, res_g)


def test_add_one():
    a = np.array([[1, 2, 3, 4, 5]])
    ta = RawTensor(a)
    res = ta + 1

    ta.to('cuda')
    res_g = ta + 1
    res_g.to('cpu')
    assert_equal(res, res_g)
    print_res(res, res_g)


def test_vec_add_broadcast():
    a = [[1, 1, 1], [3, 3, 3]]
    b = [[2, 2, 2]]
    ta = RawTensor(a)
    tb = RawTensor(b)

    res = ta + tb
    ta.to('cuda')
    tb.to('cuda')

    res_g = ta + tb
    res_g.to('cpu')
    assert_equal(res, res_g)
    print_res(res, res_g)
    

def test_vec_sub_broadcast():
    a = [[1, 1, 1], [3, 3, 3]]
    b = [[2, 2, 2]]
    ta = RawTensor(a)
    tb = RawTensor(b)

    res = ta - tb
    ta.to('cuda')
    tb.to('cuda')

    res_g = ta - tb
    res_g.to('cpu')
    assert_equal(res, res_g)
    print_res(res, res_g)
    


def test_add_vec_large():
    N = 50000000  
    a = np.ones((3, N), dtype=np.float32)
    b = np.full((3, N), 2.0, dtype=np.float32)
    c = np.full((3, N), 3.0, dtype=np.float32)
    print(a.shape)

    # ===== CPU 计算 =====
    ta = RawTensor(a)
    tb = RawTensor(b)
    tc = RawTensor(c)

    start_cpu = time.time()
    res_cpu = ta + tb + tc
    cpu_time = time.time() - start_cpu


    # ===== CUDA 计算 =====
    ta.to('cuda')
    tb.to('cuda')
    tc.to('cuda')

    start_gpu = time.time()
    res_gpu = ta + tb + tc
    gpu_time = time.time() - start_gpu

    res_gpu.to('cpu')

    # ===== 比较结果与时间 =====
    print(f"CPU time:  {cpu_time * 1000:.2f} ms")
    print(f"CUDA time: {gpu_time * 1000:.2f} ms")
    print(f"Max error: {np.abs(res_cpu.data - res_gpu.data).max()}")
    print("First 3 results:", res_gpu.data[:3])


def test_sub_vec_large():
    N = 50000000  
    a = np.ones((3, N), dtype=np.float32)
    b = np.full((3, N), 2.0, dtype=np.float32)
    c = np.full((3, N), 3.0, dtype=np.float32)
    print(a.shape)

    # ===== CPU 计算 =====
    ta = RawTensor(a)
    tb = RawTensor(b)
    tc = RawTensor(c)

    start_cpu = time.time()
    res_cpu = ta - tb - tc
    cpu_time = time.time() - start_cpu


    # ===== CUDA 计算 =====
    ta.to('cuda')
    tb.to('cuda')
    tc.to('cuda')

    start_gpu = time.time()
    res_gpu = ta - tb - tc
    gpu_time = time.time() - start_gpu

    res_gpu.to('cpu')

    # ===== 比较结果与时间 =====
    print(f"CPU time:  {cpu_time * 1000:.2f} ms")
    print(f"CUDA time: {gpu_time * 1000:.2f} ms")
    print(f"Max error: {np.abs(res_cpu.data - res_gpu.data).max()}")
    print("First 3 results:", res_gpu.data[:3])



def test_vec_mul():
    a = [[1, 1, 1], [3, 3, 3]]
    b = [5]
    # a,b = b,a
    ta = RawTensor(a)
    tb = RawTensor(b)

    res_cpu = ta * tb

    ta.to('cuda')
    tb.to('cuda')
    res_gpu = ta * tb
    res_gpu.to('cpu')

    assert_equal(res_cpu, res_gpu)
    print_res(res_cpu, res_gpu)


def test_vec_mul_broadcast():
    a = [[1, 1, 1], [3, 3, 3]]
    b = [[2, 2, 2]]
    ta = RawTensor(a)
    tb = RawTensor(b)

    res_cpu = ta * tb

    ta.to('cuda')
    tb.to('cuda')
    res_gpu = ta * tb
    res_gpu.to('cpu')

    assert_equal(res_cpu, res_gpu)
    print_res(res_cpu, res_gpu)


def test_power_large():
    N = 100_000_000  
    x = np.full((3, N), 2.0, dtype=np.float32)
    power = 3.0

    print("Input shape:", x.shape)

    # ===== CPU 计算 =====
    tx = RawTensor(x)
    start_cpu = time.time()
    res_cpu = tx ** power  # 调用 Tensor.__pow__
    cpu_time = time.time() - start_cpu
    res_cpu_np = res_cpu.data  # NumPy 结果

    # ===== CUDA 计算 =====
    tx_gpu = tx.to('cuda')
    start_gpu = time.time()
    res_gpu = tx_gpu ** power
    gpu_time = time.time() - start_gpu

    res_gpu_cpu = res_gpu.to('cpu')
    res_gpu_np = res_gpu_cpu.data

    # ===== 比较结果与时间 =====
    print(f"CPU time:  {cpu_time * 1000:.2f} ms")
    print(f"CUDA time: {gpu_time * 1000:.2f} ms")
    print(f"Max error: {np.abs(res_gpu_np - res_cpu_np).max()}")
    print("First 3 values:", res_gpu_np.ravel()[:3])


def test_sum():
    for i in range (10000, 20000):
        a = [1] * i
        print(i, len(a))
        ta = RawTensor(a)
        ta.to('cuda')
        ta = ta.sum()
        ta.to('cpu')
        assert ta.data[0] == len(a)


def test_sum_axis0():
    a0 = [1] * 12345
    a1 = [2] * 12345
    a = [a0, a1]
    na = np.array(a)
    nas = na.sum(axis=0, keepdims=False)
    print(nas[0], nas.shape)
    ta = RawTensor(a)
    ta.to('cuda')
    tb = ta.sum(axis=0, keepdims=False)
    tb.to('cpu')
    print(tb.data[0], tb.shape)

def test_sum_axis1():
    for i in range (10000, 12345):
        # print(i)
        a0 = [1] * i
        a1 = [2] * i
        a2 = [3] * i
        a = [a0, a1, a2]
        na = np.array(a)
        nas = na.sum(axis=1, keepdims=True)
        # print(nas, nas.shape)
        ta = RawTensor(a)
        ta.to('cuda')
        tb = ta.sum(axis=1, keepdims=True)
        tb.to('cpu')
        # print(tb.data, tb.shape)
        assert nas[0] == tb.data[0] and nas[1] == tb.data[1] and nas[2] == tb.data[2]


def test_ones_like():
    a = [[1, 2, 3], [4, 5, 6]]
    ta = RawTensor(a)
    ta.to('cuda')
    tb = RawTensor.ones_like(ta)
    tb.to('cpu')
    print(tb.data)
    print(tb.shape)

    na = np.array(a)
    nb = np.ones_like(na)
    print(nb)
    print(nb.shape)


def test_matmul():
    for i in range(1000):
        M = random.randint(1, 64)
        K = random.randint(1, 64)
        N = random.randint(1, 64)

        a_np = np.random.randint(-10, 10, size=(M, K), dtype=np.int32)
        b_np = np.random.randint(-10, 10, size=(K, N), dtype=np.int32)

        # CPU matmul
        ta = RawTensor(a_np)
        tb = RawTensor(b_np)
        res_cpu = ta @ tb

        # GPU matmul
        ta.to('cuda')
        tb.to('cuda')
        res_gpu = ta @ tb
        res_gpu.to('cpu')

        try:
            assert_equal(res_cpu, res_gpu)
        except AssertionError as e:
            print(f"Mismatch at case {i}: shape a={a_np.shape}, b={b_np.shape}")
            raise e


def test_transpose_matrix():
    for i in range(1000):
        M = random.randint(1, 64)
        N = random.randint(1, 64)

        a = np.random.randint(-10, 10, size=(M, N), dtype=np.int32)

        # a = [[1, 2, 3], [4, 5, 6]]
        ta = RawTensor(a)
        res_cpu = ta.T

        ta.to('cuda')
        res_gpu = ta.T
        res_gpu.to('cpu')
        
        assert_equal(res_cpu, res_gpu)