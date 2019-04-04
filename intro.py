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
        dev = kwargs['dev'] if 'dev' in kwargs else None
        print(f"{func.__name__} (dev={dev}): Duration = {timer()-start:.2f}")
    return wrapper

@time_decorator
def numpy_learn(N, D_in, H, D_out, iters, learning_rate, dev=None):
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
    print(f"loss {startingloss:.1f} -> {loss:.3g}")

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
    print(f"loss {startingloss:.1f} -> {loss:.3g}")




@time_decorator
def torch_autograd_learn(N, D_in, H, D_out, iters, learning_rate, dev="cpu"):

    dtype = torch.float
    device = torch.device(dev)

    # Create random input and output data
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)

    # Randomly initialize weights
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    startingloss = (y_pred - y).pow(2).sum()

    for t in range(iters):
        # Forward pass: compute predicted y using operations on Tensors; these
        # are exactly the same operations we used to compute the forward pass using
        # Tensors, but we do not need to keep references to intermediate values since
        # we are not implementing the backward pass by hand.
        y_pred = x.mm(w1).clamp(min=0).mm(w2)

        # Compute and print loss using operations on Tensors.
        # Now loss is a Tensor of shape (1,)
        # loss.item() gets the a scalar value held in the loss.
        loss = (y_pred - y).pow(2).sum()
#        print(t, loss.item())

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Tensors with requires_grad=True.
        # After this call w1.grad and w2.grad will be Tensors holding the gradient
        # of the loss with respect to w1 and w2 respectively.
        loss.backward()

        # Manually update weights using gradient descent. Wrap in torch.no_grad()
        # because weights have requires_grad=True, but we don't need to track this
        # in autograd.
        # An alternative way is to operate on weight.data and weight.grad.data.
        # Recall that tensor.data gives a tensor that shares the storage with
        # tensor, but doesn't track history.
        # You can also use torch.optim.SGD to achieve this.
        with torch.no_grad():
                w1 -= learning_rate * w1.grad
                w2 -= learning_rate * w2.grad

                # Manually zero the gradients after updating weights
                w1.grad.zero_()
                w2.grad.zero_()
    print(f"loss {startingloss:.1f} -> {loss:.3g}")




class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


@time_decorator
def torch_relu_learn(N, D_in, H, D_out, iters, learning_rate, dev="cpu"):
    dtype = torch.float
    device = torch.device("cpu")
    
    x = torch.randn(N, D_in, device=device, dtype=dtype)
    y = torch.randn(N, D_out, device=device, dtype=dtype)
    w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
    w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    startingloss = (y_pred - y).pow(2).sum()

    for t in range(iters):
        # To apply our Function, we use Function.apply method. We alias this as 'relu'.
        relu = MyReLU.apply

        # Forward pass: compute predicted y using operations; we compute
        # ReLU using our custom autograd operation.
        y_pred = relu(x.mm(w1)).mm(w2)

        # Compute and print loss
        loss = (y_pred - y).pow(2).sum()
#        print(t, loss.item())

        # Use autograd to compute the backward pass.
        loss.backward()

        # Update weights using gradient descent
        with torch.no_grad():
                w1 -= learning_rate * w1.grad
                w2 -= learning_rate * w2.grad

                # Manually zero the gradients after updating weights
                w1.grad.zero_()
                w2.grad.zero_()
    print(f"loss {startingloss:.1f} -> {loss:.3g}")



#%%



# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 100
learning_rate = 1e-6
iters = 5000

numpy_learn(N, D_in, H, D_out, iters, learning_rate)
torch_learn(N, D_in, H, D_out, iters, learning_rate, dev="cpu")
#torch_learn(N, D_in, H, D_out, iters, learning_rate, dev="cuda:0")
torch_autograd_learn(N, D_in, H, D_out, iters, learning_rate, dev="cpu")
torch_relu_learn(N, D_in, H, D_out, iters, learning_rate, dev="cpu")
