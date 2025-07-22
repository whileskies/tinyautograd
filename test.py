from tinyautograd import *

def test_train_cuda():
    model = MLP(3, [4, 4], 1, activation_fun=tanh)
    model.to('cuda')
    opt = SGD(model.parameters(), lr=0.05)

    x = Tensor([[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]])
    ys = Tensor([[1.0], [-1.0], [-1.0], [1.0]])
    x.to('cuda')
    ys.to('cuda')
    print(model(x).data)
    loss = mse_loss(model(x), ys)
    print(loss.data)

    for i in range(500):
        loss = mse_loss(model(x), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(i, loss.data)

    print('pred: ', model(x).data)

test_train_cuda()

