from PIL import Image
from torch.utils.data import Dataset, DataLoader
import dataset_config as config
import os
from torchvision.utils import save_image
import torch


class train_dataset(Dataset):
    def __init__(self, imgs):
        self.files = imgs
        self.len_data = len(self.files)

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        low_path = self.files[index]
        high_path = config.get_highpath(low_path)
        

        x = Image.open(low_path).convert("RGB")
        y = Image.open(high_path).convert("RGB")

        x, y = config.RandomSameCrop(x, y)
        x, y = config.apply_transforms(x,y)
        
        x = config.tfms(x)
        y = config.tfms(y)

        x = config.add_gaussian_noise_to_images(x)
        

        return x, y

class test_dataset(Dataset):
    def __init__(self, imgs):
        self.files = imgs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        
        img_path = self.files[index]
        x = Image.open(img_path).convert("RGB")
        
        x = config.tfms_test(x)

        return x

def train_loader():
    myds = train_dataset(config.get_image_files(config.TRAIN_PTH))
    loader = DataLoader(
        myds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    return loader

def test_loader():
    myds = test_dataset(config.get_image_files(config.TEST_PATH))
    loader = DataLoader(
        myds,
        batch_size=8,
        shuffle=False,
        num_workers=config.num_workers,
    )
    return loader

    




