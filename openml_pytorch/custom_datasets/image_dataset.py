
import os

import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image


class OpenMLImageDataset(Dataset):
    """
    Class representing an image dataset from OpenML for use in PyTorch.

    Methods:

        __init__(self, X, y, image_size, image_dir, transform_x=None, transform_y=None)
            Initializes the dataset with given data, image size, directory, and optional transformations.

        __getitem__(self, idx)
            Retrieves an image and its corresponding label (if available) from the dataset at the specified index. Applies transformations if provided.

        __len__(self)
            Returns the total number of images in the dataset.
    """

    def __init__(self, X, y, image_size, image_dir, transform_x=None, transform_y=None):
        self.X = X
        self.y = y
        self.image_size = image_size
        self.image_dir = image_dir
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __getitem__(self, idx):
        img_name = str(os.path.join(self.image_dir, self.X.iloc[idx, 0]))
        # hotfix for .DS_Store files
        if ".DS_Store" in img_name:
            return self.__getitem__((idx + 1) % len(self))
        
        # Open the image using PIL instead of read_image
        try:
            image = Image.open(img_name).convert("RGB")  # Ensure it's in RGB mode
        except Exception as e:
            print(f"Error opening image {img_name}: {e}")
            return self.__getitem__((idx + 1) % len(self))
        
        # Resize using PIL-based transform
        image = T.Resize((self.image_size, self.image_size))(image)
         # Convert to tensor after all PIL transformations
        image = T.ToTensor()(image)  
        # Apply additional transformations if provided
        if self.transform_x is not None:
            image = self.transform_x(image)

        if self.y is not None:
            label = self.y.iloc[idx]
            if label is not None:
                if self.transform_y is not None:
                    label = self.transform_y(label)
                return image, label
        else:
            return image

    def __len__(self):
        return len(self.X)
