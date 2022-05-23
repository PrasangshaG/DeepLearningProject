from torch.utils.data import Dataset
import os
from os.path import join, isfile
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


class GTSRB_Train(Dataset):

    def combine(self, root):
        folders = sorted(os.listdir(root))
        dfs = []
        images = []
        for d in folders:
            sub_folder = join(root, d)
            folder_images = []
            if not isfile(sub_folder):
                for s in os.listdir(sub_folder):
                    file = join(sub_folder, s)
                    if s == f"GT-{d}.csv":
                        dfs.append(pd.read_csv(file, delimiter=";"))
                    else:
                        folder_images.append(file)
                images += sorted(folder_images)
        return images, pd.concat(dfs, ignore_index=True, axis=0)

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
