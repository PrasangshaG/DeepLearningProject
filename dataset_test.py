from torch.utils.data import Dataset
import os
from os.path import join
import torch
import pandas as pd
from PIL import Image

"""
    This class is used to load the German Traffic Sign Recognition Benchmark (GTSRB) dataset.
    The dataset is composed of images of traffic signs and their labels.
    The dataset is split into training and test sets.
    The training set is used to train the model and the test set is used to evaluate the model.
    The dataset is composed of 43 classes.
"""


class GTSRB_Test(Dataset):

    def combine(self, root):
        files = sorted(os.listdir(root))
        images = []
        df = None
        for f in files:
            file = join(root, f)
            if f[-3:] == "csv":
                df = pd.read_csv(file, delimiter=';')
                df.sort_values(by=['Filename'], inplace=True)
            else:
                images.append(file)
        return images, df

    def __init__(self, root, transform=None):
        self.images, self.annotations = self.combine(root)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        y_label = torch.tensor(int(self.annotations.iloc[index, 7]))

        if self.transform:
            image = self.transform(image)

        return (image, y_label)
