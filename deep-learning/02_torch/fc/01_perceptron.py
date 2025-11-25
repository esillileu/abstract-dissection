import torch

# data
x_data = [[0., 0.], [0., 1.], [1., 0.], [1., 1.]]
# y_data = [[0. ], [0.], [0.], [0.]]  # false
y_data = [[0.], [0.], [0.], [1.]] # and
# y_data = [[0.], [0.], [1.], [1.]] # not
# y_data = [[0.], [1.], [1.], [1.]] # or
# y_data = [[1.], [1.], [1.], [1.]] # true
# y_data = [[0.], [1.], [1.], [0.]] # xor - impossible

x = torch.tensor(x_data)  # (b, 2)
y = torch.tensor(y_data)

# model
w = torch.rand((2, 1))
b = torch.rand((1, ))
step_func = lambda x : (x > 0).float()

model = lambda x: step_func(x @ w + b)

# train 
lr = 0.1 
for _ in range(100):
    y_hat = model(x)
    e = y - y_hat
    dw = lr * ( x.T @ e )
    db = lr * e.sum()
    if (dw == 0).all() and (db == 0).all():
        break
    w = w + dw
    b = b + db


# test
print(model(x))