from PIL import Image
from torch.utils.data import Dataset, DataLoader
import dataset_config as config
import os
from torchvision.utils import save_image
import torch
import random

class train_dataset_for_cue_model(Dataset):
    def __init__(self, imgs):
        self.files = imgs

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        low_path = self.files[index]
        high_path = config.get_highpath(low_path)
        

        x = Image.open(low_path).convert("RGB")
        y = Image.open(high_path).convert("RGB")

        x, y = config.RandomSameCrop(x, y)
        x, y = config.apply_transforms(x,y)
        
        x = config.tfms(x)
        LR = config.low_tfms(y)
        edge = config.edge_tfms(y)
        y = config.tfms(y)

        x = config.add_gaussian_noise_to_images(x)
        

        return x, y, LR, edge

def train_loader_cue():
    myds = train_dataset_for_cue_model(config.get_image_files(config.TRAIN_PTH))
    loader = DataLoader(
        myds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    return loader

#--------------------------------------------------
class train_dataset_for_CBDFE_model(Dataset):
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

        x_pos = config.weak_tfms(x)
        #----------------------select negative sample------------------------------------
        idx = random.choice([idx for idx in range(len(self.files)) if index != idx])
        path = os.path.join(low_path, self.files[idx]) 
        img = Image.open(path).convert("RGB")
        x_neg = config.weak_tfms(img) #[config.weak_tfms(img) for _ in range(config.K_augmented)]
        #--------------------------------------------------------------------------------

        x, y = config.tfms(x), config.tfms(y)    

        return x, y, x_pos, x_neg

def train_loader_CBDFE():
    myds = train_dataset_for_CBDFE_model(config.get_image_files(config.TRAIN_PTH))
    loader = DataLoader(
        myds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    return loader

#--------------------------------------------------

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

if __name__ == "__main__":
    
    # loader = test_loader()
    
    # x = next(iter(loader))
    # print(x.shape)

    # save_image(x*.5+.5, "outcomes/datasample/x_test.png")

    loader = train_loader()
    
    x, y = next(iter(loader))
    print(x.shape)
    print(y.shape)

    concat_cover = torch.cat((x*.5+.5, y*.5+.5), 2)

    save_image(concat_cover, "outcomes/datasample/lol.png")
    




