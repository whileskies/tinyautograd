# TinyAutoGrad

TinyAutoGrad is an automatic differentiation engine and neural network library inspired by [Micrograd](https://github.com/karpathy/micrograd), with optional CUDA support.

---

## 🚀 Features

- ⚙️ Automatic differentiation (forward & backward)
- 🧱 Simple `Tensor` class with NumPy-style operations
- 🧠 Neural network support with MLP example
- 💻 CPU and ⚡ CUDA backend support
- 🧪 Built-in testing with `pytest`

---

## 🛠️ Getting Started

### CPU (Default)

Run the MNIST example on CPU:

```bash
python -m samples.mnist.mnist
```

---

### CUDA (Experimental)

Ensure CUDA toolkit (`nvcc`) is installed:

```bash
# Compile CUDA backend
nvcc -shared -o libops.so tinyautograd/ops.cu -Xcompiler -fPIC -lcublas

# Run test.py
python test.py

# Run MNIST with CUDA backend
python -m samples.mnist.mnist_cuda
```

> ⚠️ Make sure `libops.so` is in the current directory or Python load path.

---

## 🧪 Running Tests

Run all unit tests:

```bash
pytest tinyautograd/ -s -v --cache-clear
```

Or a specific test:

```bash
pytest tinyautograd/test_rawtensor.py -s -v --cache-clear
```

---

## 📂 Project Structure

```
.
├── README.md
├── libops.so
├── test.py
├── samples
│   └── mnist
│       ├── mnist.py
│       ├── mnist_cuda.py
│       ├── train-images-idx3-ubyte.gz
│       └── train-labels-idx1-ubyte.gz
├── tinyautograd
│   ├── __init__.py
│   ├── functional.py
│   ├── nn.py
│   ├── ops.cu
│   ├── optim.py
│   ├── rawtensor.py
│   ├── tensor.py
│   ├── test_nn.py
│   ├── test_rawtensor.py
│   └── test_tensor.py
```

---

## 📄 License

MIT