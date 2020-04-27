import torch as t
from trainer import Trainer
import sys
import torchvision as tv
from model import resnet

#epoch = int(sys.argv[1])
epoch = 9
#TODO: Enter your model here
model = resnet.ResNet()
crit = t.nn.BCEWithLogitsLoss()
trainer = Trainer(model, crit)
trainer.restore_checkpoint(epoch)
trainer.save_onnx('checkpoint_{:03d}.onnx'.format(epoch))
