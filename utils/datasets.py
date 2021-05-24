import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class TrianValDataset(Dataset):
    def __init__(self, img_df, transform=None):
        self.img_df = img_df
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_df.iloc[index]["path"]).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        return img, torch.from_numpy(np.array(self.img_df.iloc[index]["Type"]))

    def __len__(self):
        return len(self.img_df)


class TestDataset(Dataset):
    def __init__(self, img_df, transform=None):
        self.img_path = img_df["path"]
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img = Image.open(self.img_path[index])
        if self.transform is not None:
            img = self.transform(img)
        return img
