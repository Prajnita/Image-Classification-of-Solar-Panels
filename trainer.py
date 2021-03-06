import torch as t
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm
#from evaluation import create_evaluation
#from ignite.metrics import Accuracy
import shutil
from model import resnet
from torch import onnx


class Trainer:

    def __init__(self,
                 model,             # Model to be trained.
                 crit,
                 # Loss function
                 optim = None,         # Optimiser
                 train_dl = None,      # Training data set
                 val_test_dl = None,   # Validation (or test) data set
                 cuda = True,          # Whether to use the GPU
                 early_stopping_cb = None): # The stopping criterion.
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = True
        self._early_stopping_cb = early_stopping_cb

        if self._cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()

    def save_checkpoint(self, epoch):
    #def save_checkpoint(self, epoch, filename='checkpoint.pth.tar'):
        #t.save(epoch,filename)
        t.save({'state_dict': self._model.state_dict()}, 'checkpoints/checkpoint_{:03d}.ckp'.format(epoch))

    def restore_checkpoint(self, epoch_n):
        ckp = t.load('checkpoints/checkpoint_{:03d}.ckp'.format(epoch_n), 'cuda' if self._cuda else None)
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        y = self._model(x)
        t.onnx.export(m,                 # model being run
              x,                         # model input (or a tuple for multiple inputs)
              fn,                        # where to save the model (can be a file or file-like object)
              export_params=True,        # store the trained parameter weights inside the model file
              opset_version=10,          # the ONNX version to export the model to
              do_constant_folding=True,  # whether to execute constant folding for optimization
              input_names = ['input'],   # the model's input names
              output_names = ['output'], # the model's output names
              dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            'output' : {0 : 'batch_size'}})

    def train_step(self, x, y):
        # perform following steps:
        # -reset the gradients
        self._optim.zero_grad()
        # -propagate through the network
        output = self._model(x)
        # -calculate the loss
        loss = self._crit(output, y)
        # -compute gradient by backward propagation
        loss.backward()
        # -update weights
        self._optim.step()
        # -return the loss
        return loss


    def val_test_step(self, x, y):
        # predict
        # propagate through the network and calculate the loss and predictions
        predictions = self._model(x)
        loss = self._crit(predictions, y)
        # return the loss and the predictions
        return loss, predictions

    def train_epoch(self):
        # TODO: set training mode
        self._model.train()
        # iterate through the training set
        # transfer the batch to "cuda()" -> the gpu if a gpu is given
        # perform a training step
        # calculate the average loss for the epoch and return it

        running_loss = 0.0
        for x, y in self._train_dl:
            if self._cuda is not False:
                self._model.cuda()
                x = x.cuda()
                y = y.cuda()
            loss = self.train_step(x, y)
            running_loss += loss
        return running_loss/len(self._train_dl)
        #TODO

    def val_test(self):
        # set eval mode
        self._model.eval()
        # disable gradient computation
        t.no_grad()
        # iterate through the validation set
        # transfer the batch to the gpu if given
        # perform a validation step
        # save the predictions and the labels for each batch
        # calculate the average loss and average metrics of your choice. You might want to calculate these metrics in designated functions
        # return the loss and print the calculated metrics
        running_loss = 0.0

        acc_crack = 0
        acc_inactive = 0
        for x, y in self._val_test_dl:
            if self._cuda is not False:
                self._model.cuda()
                x = x.cuda()
                y = y.cuda()
            loss, predictions = self.val_test_step(x, y)
            running_loss += float(loss)
        return running_loss
        #TODO

    def fit(self, epochs=40):
        assert self._early_stopping_cb is not None or epochs > 0
        # create a list for the train and validation losses, and create a counter for the epoch 
        train_losses = []
        val_losses = []
        epoch_counter = 0
        #TODO
        while True:
            # stop by epoch number
            if epoch_counter == epochs:
                break
            # train for a epoch and then calculate the loss and metrics on the validation set
            train_loss = self.train_epoch()
            print(epoch_counter)
            print('train loss:')
            print(train_loss)
            val_loss = self.val_test()
            # append the losses to the respective lists
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print('val loss:')
            print(val_loss)
            # use the save_checkpoint function to save the model for each epoch
            Trainer.save_checkpoint(self, epoch_counter)
            #Trainer.save_onnx(self, exp)
            #self.save_checkpoint({'epoch_counter': epoch_counter + 1,'state_dict': self._model.state_dict()})
            #self.save_checkpoint(epoch_counter)
            # check whether early stopping should be performed using the early stopping callback and stop if so
            if self._early_stopping_cb.step(train_losses):
                break
            #print("No early stopping")
            epoch_counter += 1
            #print(epoch_counter)

        # return the loss lists for both training and validation
        return train_losses, val_losses
        #TODO
                    
        
        
        