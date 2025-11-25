import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from function import relu, nll_loss

# data
transform = transforms.ToTensor()
dataset_train = datasets.MNIST(
    root=".data", train=True, download=True, transform=transform
)
dataset_test = datasets.MNIST(
    root=".data", train=False, download=True, transform=transform
)
loader_train = DataLoader(dataset_train, 256, True)
loader_test = DataLoader(dataset_test, 128, True)

# model
# layer 1
norm = 0.01
w1 = torch.randn((28 * 28, 128)) * norm
w1.requires_grad_(True)
b1 = torch.zeros((128,), requires_grad=True)
# layer 2
w2 = torch.randn((128, 32)) * norm
w2.requires_grad_(True)
b2 = torch.zeros((32,), requires_grad=True)

# layer 3
w3 = torch.randn((32, 10)) * norm
w3.requires_grad_(True)
b3 = torch.zeros((10,), requires_grad=True)

params = [w1, w2, w3, b1, b2, b3]

active_fn = relu.apply

layer = lambda x, w, b: active_fn(x @ w + b)
model = lambda x: layer(layer(x.view(x.size(0), -1), w1, b1), w2, b2) @ w3 + b3

criteria = nll_loss

# train
lr = 0.1
epoch = 20

ebar = tqdm(range(epoch), total=epoch, desc="outer")

for i in ebar:
    train_loss = 0
    bbar = tqdm(loader_train, total=len(loader_train), desc="inner", leave=False)
    for x, y in bbar:
        y_pred = model(x)
        loss = criteria(y_pred, y)
        loss.backward()

        with torch.no_grad():
            for p in params:
                p -= lr * p.grad
                p.grad.zero_()

        train_loss += loss.item()

    train_loss /= len(loader_train)
    ebar.postfix = f"Epoch [{i + 1}/{epoch}], Loss: {loss.item():.6f}"

with torch.no_grad():
    print(model(x[0]), y[0].item())
