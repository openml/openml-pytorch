
import torch
from torch.utils.data import Dataset


class OpenMLTabularDataset(Dataset):
    """
    OpenMLTabularDataset

    A custom dataset class to handle tabular data from OpenML (or any similar tabular dataset).
    It encodes categorical features and the target column using LabelEncoder from sklearn.

    Methods:
        __init__(X, y) : Initializes the dataset with the data and the target column.
                         Encodes the categorical features and target if provided.

        __getitem__(idx): Retrieves the input data and target value at the specified index.
                          Converts the data to tensors and returns them.

        __len__(): Returns the length of the dataset.
    """

    def __init__(self, X, y):
        self.data = X
        # self.target_col_name = target_col
        for col in self.data.select_dtypes(include=["object", "category"]):
            # convert to float
            self.data[col] = self.data[col].astype("category").cat.codes
        self.label_mapping = None

        self.y = y

    def __getitem__(self, idx):
        # x is the input data, y is the target value from the target column
        x = self.data.iloc[idx, :]
        x = torch.tensor(x.values.astype("float32"))
        if self.y is not None:
            y = self.y[idx]
            y = torch.tensor(y)
            return x, y
        else:
            return x

    def __len__(self):
        return len(self.data)
