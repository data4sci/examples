# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
#%% 
import numpy as np
import torch
from timeit import default_timer as timer


print('ok')

#%%

def time_decorator(func):
    def wrapper(*args, **kwargs):
        start = timer()
        func(*args, **kwargs)
        print(f"Duration = {timer()-start}")
    return wrapper

@time_decorator
def numpy_learn(N, D_in, H, D_out, iters, learning_rate):
    x = np.random.randn(N, D_in)
    y = np.random.randn(N, D_out)
    w1 = np.random.randn(D_in, H)
    w2 = np.random.randn(H, D_out)

    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)
    startingloss = np.square(y_pred - y).sum()
    for t in range(iters):
        # Forward pass: compute predicted y
        h = x.dot(w1)
        h_relu = np.maximum(h, 0)
        y_pred = h_relu.dot(w2)

        # Compute and print loss
        loss = np.square(y_pred - y).sum()
#        print(t, loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.T.dot(grad_y_pred)
        grad_h_relu = grad_y_pred.dot(w2.T)
        grad_h = grad_h_relu.copy()
        grad_h[h < 0] = 0
        grad_w1 = x.T.dot(grad_h)

        # Update weights
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
    print(f"loss {startingloss} -> {loss}")

@time_decorator
def torch_learn(N, D_in, H, D_out, iters, learning_rate, dev="cpu"):

    dtype = torch.float
    device = torch.device(dev)

    # Create random input and output data
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    # Randomly initialize weights
    w1 = torch.randn(D_in, H, device=device, dtype=dtype)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype)

    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)
    startingloss = (y_pred - y).pow(2).sum().item()

    for t in range(iters):
        # Forward pass: compute predicted y
        h = x.mm(w1)
        h_relu = h.clamp(min=0)
        y_pred = h_relu.mm(w2)

        # Compute and print loss
        loss = (y_pred - y).pow(2).sum().item()
#        print(t, loss)

        # Backprop to compute gradients of w1 and w2 with respect to loss
        grad_y_pred = 2.0 * (y_pred - y)
        grad_w2 = h_relu.t().mm(grad_y_pred)
        grad_h_relu = grad_y_pred.mm(w2.t())
        grad_h = grad_h_relu.clone()
        grad_h[h < 0] = 0
        grad_w1 = x.t().mm(grad_h)

        # Update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
    print(f"loss {startingloss} -> {loss}")

#%%



# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 100
learning_rate = 1e-6
iters = 5000

numpy_learn(N, D_in, H, D_out, iters, learning_rate)
torch_learn(N, D_in, H, D_out, iters, learning_rate, "cpu")
#torch_learn(N, D_in, H, D_out, iters, learning_rate, "cuda:0")
