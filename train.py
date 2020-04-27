import torch as t
import torchvision as tv
from data import get_train_dataset, get_validation_dataset
from stopping import EarlyStoppingCallback
from trainer import Trainer
from matplotlib import pyplot as plt
import numpy as np
from model import resnet

# set up data loading for the training and validation set using t.utils.data.DataLoader and the methods implemented in data.py
train_dl = t.utils.data.DataLoader(get_train_dataset(), batch_size=50)
test_dl = t.utils.data.DataLoader(get_validation_dataset(), batch_size=20)
# set up your model
model = resnet.ResNet()
# set up loss (you can find preimplemented loss functions in t.nn) use the pos_weight parameter to ease convergence
loss = t.nn.MultiLabelSoftMarginLoss()
# set up optimizer (see t.optim);
optimizer = t.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# initialize the early stopping callback implemented in stopping.py and create a object of type Trainer
early_stopping = EarlyStoppingCallback(5)
trainer = Trainer(model, loss, optim=optimizer, train_dl=train_dl, val_test_dl=test_dl, cuda=True,
                  early_stopping_cb=early_stopping)

# go, go, go... call fit on trainer
res = trainer.fit()

# plot the results
plt.plot(np.arange(len(res[0])), res[0], label='train loss')
plt.plot(np.arange(len(res[1])), res[1], label='val loss')
plt.yscale('log')
plt.legend()
plt.savefig('losses.png')