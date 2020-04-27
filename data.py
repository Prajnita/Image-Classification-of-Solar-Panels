from torch.utils.data import Dataset
import torch
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision as tv
import imageio
import operator

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]
path = 'train.csv'
data_frame = pd.read_csv(path, sep=';')
split = 0.3
X_train, X_test = train_test_split(data_frame, test_size=split, random_state=42)


class ChallengeDataset(Dataset):
    # TODO implement the Dataset class according to the description
    def __init__(self, flag, path, split, transforms=tv.transforms.Compose([tv.transforms.ToTensor()])):
        # TODO: perform split here?
        #self.data_frame = pd.read_csv(path, sep=';')
        #X_train, X_test = split(self.data_frame, test_size=split, random_state=42)
        if flag == "train":
            self.data_frame = X_train
        else:
            self.data_frame = X_test
        #self.flag = flag
        self.split = split
        self.transforms = transforms

    def __getitem__(self, index):
        # convert dataframe to numpy array for easier indexing
        data_frame = self.data_frame.to_numpy()
        # get image and labels
        labels = data_frame[index][2:]
        labels = labels.astype(int)
        path = data_frame[index][0]
        img = imageio.imread(path)
        # transform to rgb and follow transformations in transform object
        img = gray2rgb(img)
        img = self.transforms(img)
        labels = torch.from_numpy(labels)
        return img, labels

    def __len__(self):
        return len(self.data_frame.index)

    def pos_weight(self):
        data_frame = self.data_frame.to_numpy()
        data_frame = data_frame.to_numpy()
        weights = []
        for i in range(1, data_frame.shape[1]):
            weight = np.count_nonzero(data_frame[:, i])
            weight = 1 / weight
            weights.append(weight)
        return torch.Tensor(weights)


def get_train_dataset():
    flag = "train"
    path = "train.csv"

    # TODO: how should we initialize the split and where should we perform the split
    split = 0.7
    transforms = tv.transforms.Compose([
        tv.transforms.RandomRotation(90),
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(train_mean, train_std, inplace=True)
    ])
    return ChallengeDataset(flag, path, split, transforms)


# this needs to return a dataset *without* data augmentation!
def get_validation_dataset():
    flag = "val"
    path = "train.csv"
    # TODO: how should we initialize the split and where should we perform the split
    split = 0.3
    transforms = tv.transforms.Compose([
        tv.transforms.ToPILImage(),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(train_mean, train_std, inplace=True)
    ])
    return ChallengeDataset(flag, path, split, transforms)


