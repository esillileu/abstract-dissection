import torch
from tqdm import tqdm

# data
x_data = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
y_data = [[0.0], [1.0], [1.0], [0.0]]  # xor - impossible at perceptron

x = torch.tensor(x_data)  # (b, 2)
y = torch.tensor(y_data)

# model
# layer 1
w1 = torch.rand((2, 2), requires_grad=True)
b1 = torch.rand((2,), requires_grad=True)
# layer 2
w2 = torch.rand((2, 1), requires_grad=True)
b2 = torch.rand((1,), requires_grad=True)
params = [w1, w2, b1, b2]

active_fn = torch.sigmoid

model = lambda x: active_fn(active_fn(x @ w1 + b1) @ w2 + b2)
criteria = lambda y_pred, y: ((y_pred - y) ** 2).mean()


# train
lr = 0.1
epoch = 100000
pbar = tqdm(range(epoch), total=epoch)
for i in pbar:
    y_pred = model(x)
    loss = criteria(y_pred, y)
    loss.backward()

    with torch.no_grad():
        for p in params:
            p -= lr * p.grad
            p.grad.zero_()

    if (i + 1) % 200 == 0:
        pbar.postfix = f"Epoch [{i + 1}/{epoch}], Loss: {loss.item():.6f}"

with torch.no_grad():
    print(model(x))
