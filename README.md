# tinyautograd
TinyAutoGrad is an automatic differentiation engine and neural network library inspired by Micrograd and GPT, with CUDA support.

# todo
nvcc -shared -o tinyautograd/libops.so tinyautograd/ops.cu  -Xcompiler -fPIC