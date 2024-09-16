import os
from typing import Any
import pandas as pd
from sklearn import preprocessing
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
# from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor, Lambda
import torchvision.transforms as T

class OpenMLImageDataset(Dataset):
    def __init__(self,image_size, annotations_df, img_dir, transform=None, target_transform=None):
        self.img_labels = annotations_df
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_size = image_size
        self.has_labels = 'encoded_labels' in annotations_df.columns

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        try:
            image = read_image(img_path)
        except RuntimeError as error:
            # print(f"Error loading image {img_path}: {error}")
            # Use a default image        
            # from .config import image_size
            image = torch.zeros((3, self.image_size, self.image_size), dtype=torch.uint8)
            
        # label = self.img_labels.iloc[idx, 1]
        if self.transform is not None:
            image = self.transform(image)
            image = image.float()

        if self.has_labels:
            label = self.img_labels.iloc[idx, 1]
            return image, label
        else:
            return image

class OpenMLTabularDataset(Dataset):
    def __init__(self, annotations_df, y):
        self.data = annotations_df
        # self.target_col_name = target_col
        self.label_mapping = None

        self.label_mapping = preprocessing.LabelEncoder()
        try:
            self.y = self.label_mapping.fit_transform(y)
        except ValueError:
            self.y = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # x is the input data, y is the target value from the target column
        x = self.data.iloc[idx, :]
        x = torch.tensor(x.values.astype('float32'))
        if self.y is not None:
            y = self.y[idx]
            y = torch.tensor(y)
            return x, y
        else:
            return x
            