# tinyautograd
TinyAutoGrad is an automatic differentiation engine and neural network library inspired by [Micrograd](https://github.com/karpathy/micrograd), with CUDA support.

# cpu
python -m samples.mnist.mnist 

# cuda 🚧
nvcc -shared -o libops.so tinyautograd/ops.cu  -Xcompiler -fPIC

python -m samples.mnist.mnist-cuda