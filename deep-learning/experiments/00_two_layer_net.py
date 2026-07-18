from mlprosection.optim.SGD import SGD
from mlprosection.trainer import ForwardTrainer
from mlprosection.datasets import load_mnist
from mlprosection.nn.initailizer import he_normal_
from mlprosection.optim.transform import L2Regularization
from mlprosection.nn.layers.criterion import SoftmaxWithLoss
from mlprosection.nn.model.test import TwoLayerNet, SimpleCNN

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, gpu=True)

model = SimpleCNN().gpu()

model.layers[4].reset_weight(he_normal_)
model.layers[6].reset_weight(he_normal_)
print(model.forward(x_train[10:20]).argmax(axis=1).data, t_train[10:20].data)

criterion = SoftmaxWithLoss().gpu()
optimizer = SGD(model.named_parameters(), pre_step_hooks=[L2Regularization()])

trainer = ForwardTrainer(model, criterion, optimizer, 5, 64, 30)
trainer.fit(x_train[:], t_train[:], x_test[:], t_test[:])

print(model.forward(x_train[10:20]).argmax(axis=1).data, t_train[10:20].data)
model.save_params_npz("experiment/param/cnn.npz")
