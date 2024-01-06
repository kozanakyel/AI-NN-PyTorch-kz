import os
from pathlib import Path

from config import *
config_chapter0()
from plots.chapter0 import *
config_chapter1()
from plots.chapter1 import *

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import torch
import torch.optim as optim
import torch.nn as nn
# from torchviz import make_dot



true_b = 1
true_w = 2
N = 100

# data genration
np.random.seed(42)
x = np.random.rand(N, 1)
epsilon = (.1 * np.random.randn(N, 1))
y = true_b + true_w * x + epsilon

# train-validation
idx = np.arange(N)
np.random.shuffle(idx)
train_idx = idx[:int(N * .8)]
val_idx = idx[int(N * .8):]
x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

# fig, ax = figure1(x_train, y_train, x_val, y_val)
# fig.savefig(Path(os.getcwd(), 'image_res', 'fr.png'))

np.random.seed(42)
b = np.random.randn(1)
w = np.random.randn(1)

print(f'b and w: {b}, {w}')

yhat = b + w * x_train
# fig, ax = figure2(x_train, y_train, b, w)
# fig.savefig(Path(os.getcwd(), 'image_res', 'fr.png'))

error = yhat - y_train
loss = (error ** 2).mean()
print('loss: ', loss)

b_range = np.linspace(true_b - 3, true_b + 3, 101)
w_range = np.linspace(true_w - 3, true_w + 3, 101)
bs, ws = np.meshgrid(b_range, w_range)
print("bs ws shapes", bs.shape, ws.shape)


all_predictions = np.apply_along_axis(
    func1d=lambda x: bs + ws * x,
    axis=1,
    arr=x_train
)
all_labels = y_train.reshape(-1, 1, 1)
all_errors = (all_predictions - all_labels)
all_losses = (all_errors ** 2).mean(axis=0)
print('predictions shape', all_predictions.shape)
print('labels shape', all_labels.shape)
print('errors shape', all_errors.shape)
print('loss shape', all_losses.shape)
# fig, ax = figure5(x_train, y_train, b, w, bs, ws, all_losses)
# fig.savefig(Path(os.getcwd(), 'image_res', 'fr.png'))

# compute Gradients
b_grad = 2 * error.mean()
w_grad = 2 * (x_train*error).mean()
print("grads b, w: ", b_grad, w_grad)

# update paramaetres with learning rate ETA
lr = 0.1
b = b - lr*b_grad
w = w - lr*w_grad

# Preproccesing steps like StandartScaler must be performed After the train-validation test split
scaler = StandardScaler(with_std=True, with_mean=True)
scaler.fit(x_train)

scaled_x_train = scaler.transform(x_train)
scaled_x_val = scaler.transform(x_val)

scaled_b_range = np.linspace(-1, 5, 101)
scaled_w_range = np.linspace(-2.4, 3.6, 101)
scaled_bs, scaled_ws = np.meshgrid(scaled_b_range, scaled_w_range)

######### Simple Linear Regressions

# np.random.seed(42)
# b = np.random.randn(1)
# w = np.random.randn(1)
#
# lr = .1
# n_epochs = 1000
#
# for epoch in range(n_epochs):
#     yhat = b + w*x_train   # forward propagation
#
#     error = yhat - y_train      # computing loss
#     loss = (error**2).mean()
#
#     b_grad = 2*error.mean()         # computing gradients
#     w_grad = 2*(x_train*error).mean()
#
#     b = b - lr*b_grad           # updating parameters
#     w = w - lr*w_grad
#
# print(b, w)

#### with sklearn linear regression
linr = LinearRegression()
linr.fit(x_train, y_train)
print(linr.intercept_, linr.coef_[0])

"""
PYTORCH tensor space
scalar : zero dimension    // has empty shape for pytorch
vector : one dimension
matrix : 2 dimension
tensor : 3 or more dimension

.size() and .shape works....
for reshape:    .view() -preferred! and   .reshape()
"""
scalar = torch.tensor(3.14159)
vector = torch.tensor([1,2,3])
matrix = torch.ones((2,3), dtype=torch.float)
tensor = torch.randn((2,3,4), dtype=torch.float)

# print(f's: {scalar}\n v: {vector}\n m: {matrix}\n t: {tensor}\n')

same_matrix = matrix.view(1,6)
same_matrix[0, 1] = 2.

# for creating a new tensor use .clone or .new_tensor
# different_matrix = matrix.new_tensor(matrix.view(1, 6))
# different_matrix[0, 1] = 3.

another_matrix = matrix.view(1, 6).clone().detach()  # detach remove from computaional graph
another_matrix[0, 1] = 4.

# print(same_matrix, different_matrix, another_matrix)

x_train_tensor = torch.as_tensor(x_train)
float_tensor = x_train_tensor.float()
# print(x_train.dtype, x_train_tensor.dtype)

# both numpy array and tensor modfified  !!! WTf
dummy_array = np.array([1, 2, 3])
dummy_tensor = torch.as_tensor(dummy_array)
# Modifies the numpy array
dummy_array[1] = 0
# Tensor gets modified too...
# print(dummy_tensor)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

n_cudas = torch.cuda.device_count()
for i in range(n_cudas):
    print(torch.cuda.get_device_name(i))
gpu_tensor = torch.as_tensor(x_train).to(device)

x_train_tensor = torch.as_tensor(x_train).float().to(device)
y_train_tensor = torch.as_tensor(y_train).float().to(device)

print(type(x_train), type(x_train_tensor), x_train_tensor.type())

