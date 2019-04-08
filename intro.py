# https://pytorch.org/tutorials/beginner/pytorch_with_examples.html
#%% 
import numpy as np
import torch
import tensorflow as tf
from timeit import default_timer as timer
import random

print('imports ok')

#%%

def time_decorator(func):
    def wrapper(*args, **kwargs):
        start = timer()
        func(*args, **kwargs)
        dev = kwargs['dev'] if 'dev' in kwargs else None
        print(f"{func.__name__} (dev={dev}): Duration = {timer()-start:.2f}\n")
    return wrapper

@time_decorator
def numpy_learn(N, D_in, H, D_out, iters, threshold, learning_rate, dev=None):
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
        if loss/startingloss < threshold:
                break
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
    print(f"loss {startingloss:.1f} -> {loss:.3g}; ratio={100*loss/startingloss:.2g}%; iters={t}")

@time_decorator
def torch_learn(N, D_in, H, D_out, iters, threshold, learning_rate, dev="cpu"):

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
        if loss/startingloss < threshold:
                break
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
    print(f"loss {startingloss:.1f} -> {loss:.3g}; ratio={100*loss/startingloss:.2g}%; iters={t}")


@time_decorator
def torch_autograd_learn(N, D_in, H, D_out, iters, threshold, learning_rate, dev="cpu"):

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
        if loss/startingloss < threshold:
                break
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
    print(f"loss {startingloss:.1f} -> {loss:.3g}; ratio={100*loss/startingloss:.2g}%; iters={t}")




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
def torch_relu_learn(N, D_in, H, D_out, iters, threshold, learning_rate, dev="cpu"):
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
        if loss/startingloss < threshold:
                break
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
    print(f"loss {startingloss:.1f} -> {loss:.3g}; ratio={100*loss/startingloss:.2g}%; iters={t}")



@time_decorator
def tf_learn(N, D_in, H, D_out, iters, threshold, learning_rate, dev="cpu"):
    # First we set up the computational graph:
    # Create placeholders for the input and target data; these will be filled
    # with real data when we execute the graph.
    x = tf.placeholder(tf.float32, shape=(None, D_in))
    y = tf.placeholder(tf.float32, shape=(None, D_out))
    # Create Variables for the weights and initialize them with random data.
    # A TensorFlow Variable persists its value across executions of the graph.
    w1 = tf.Variable(tf.random_normal((D_in, H)))
    w2 = tf.Variable(tf.random_normal((H, D_out)))
    # Forward pass: Compute the predicted y using operations on TensorFlow Tensors.
    # Note that this code does not actually perform any numeric operations; it
    # merely sets up the computational graph that we will later execute.
    h = tf.matmul(x, w1)
    h_relu = tf.maximum(h, tf.zeros(1))
    y_pred = tf.matmul(h_relu, w2)
    # Compute loss using operations on TensorFlow Tensors
    loss = tf.reduce_sum((y - y_pred) ** 2.0)
    # Compute gradient of the loss with respect to w1 and w2.
    grad_w1, grad_w2 = tf.gradients(loss, [w1, w2])
    # Update the weights using gradient descent. To actually update the weights
    # we need to evaluate new_w1 and new_w2 when executing the graph. Note that
    # in TensorFlow the the act of updating the value of the weights is part of
    # the computational graph; in PyTorch this happens outside the computational
    # graph.
    new_w1 = w1.assign(w1 - learning_rate * grad_w1)
    new_w2 = w2.assign(w2 - learning_rate * grad_w2)
    # Now we have built our computational graph, so we enter a TensorFlow session to
    # actually execute the graph.
    with tf.Session() as sess:
        # Run the graph once to initialize the Variables w1 and w2.
        sess.run(tf.global_variables_initializer())
        # Create numpy arrays holding the actual data for the inputs x and targets
        # y
        x_value = np.random.randn(N, D_in)
        y_value = np.random.randn(N, D_out)
        startingloss = None
        for t in range(iters):
                # Execute the graph many times. Each time it executes we want to bind
                # x_value to x and y_value to y, specified with the feed_dict argument.
                # Each time we execute the graph we want to compute the values for loss,
                # new_w1, and new_w2; the values of these Tensors are returned as numpy
                # arrays.
                loss_value, _, _ = sess.run([loss, new_w1, new_w2],
                                        feed_dict={x: x_value, y: y_value})
                if not startingloss:
                        startingloss = loss_value
                if loss_value/startingloss < threshold:
                        break
        #        print(loss_value)
    print(f"loss {startingloss:.1f} -> {loss_value:.3g}; ratio={100*loss_value/startingloss:.2g}%; iters={t}")









class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """
        In the constructor we construct three nn.Linear instances that we will use
        in the forward pass.
        """
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        """
        For the forward pass of the model, we randomly choose either 0, 1, 2, or 3
        and reuse the middle_linear Module that many times to compute hidden layer
        representations.

        Since each forward pass builds a dynamic computation graph, we can use normal
        Python control-flow operators like loops or conditional statements when
        defining the forward pass of the model.

        Here we also see that it is perfectly safe to reuse the same Module many
        times when defining a computational graph. This is a big improvement from Lua
        Torch, where each Module could be used only once.
        """
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred

@time_decorator
def torch_final_learn(N, D_in, H, D_out, iters, threshold, learning_rate, dev="cpu"):

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)
    model = DynamicNet(D_in, H, D_out)
    
    # Construct our loss function and an Optimizer. Training this strange model with
    # vanilla stochastic gradient descent is tough, so we use momentum
    criterion = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    startingloss = None
    for t in range(iters):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)
    
        # Compute and print loss
        loss = criterion(y_pred, y)
        if not startingloss:
            startingloss = loss
        if loss/startingloss < threshold:
                break
#        print(t, loss.item())
    
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"loss {startingloss:.1f} -> {loss:.3g}; ratio={100*loss/startingloss:.2g}%; iters={t}")




#%%



# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 100
learning_rate = 1e-6
iters = 50000
threshold = 0.00001

numpy_learn(N, D_in, H, D_out, iters, threshold, learning_rate)
torch_learn(N, D_in, H, D_out, iters, threshold, learning_rate, dev="cpu")
#torch_learn(N, D_in, H, D_out, iters, threshold, learning_rate, dev="cuda:0")
torch_autograd_learn(N, D_in, H, D_out, iters, threshold, learning_rate, dev="cpu")
torch_relu_learn(N, D_in, H, D_out, iters, threshold, learning_rate, dev="cpu")
tf_learn(N, D_in, H, D_out, iters, threshold, learning_rate)
torch_final_learn(N, D_in, H, D_out, iters, threshold, learning_rate, dev="cpu")
