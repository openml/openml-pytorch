import torch
from torch.utils.data import Dataset


class OpenMLTabularDataset(torch.utils.data.Dataset):
    """
    Class representing a tabular dataset from OpenML for use in PyTorch.

    Methods:

        __init__(self, X, y, transform_x=None, transform_y=None)
            Initializes the dataset with given data and optional transformations.

        __getitem__(self, idx)
            Retrieves a sample and its corresponding label (if available) from the dataset at the specified index. Applies transformations if provided.

        __len__(self)
            Returns the total number of samples in the dataset.
    """

    def __init__(self, X, y, transform_x=None, transform_y=None):
        self.X = X
        self.y = y
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __getitem__(self, idx):
        sample = self.X.iloc[idx]
        if self.transform_x:
            sample = self.transform_x(sample)
        if self.y is not None:
            label = self.y.iloc[idx]
            if self.transform_y:
                label = self.transform_y(label)
            return sample, label
        return sample

    def __len__(self):
        return len(self.X)
