import os
import torchvision.transforms as transforms
import glob
import random
import torchvision.transforms.functional as TF
import torch
from PIL import ImageFilter
from pathlib import Path


# TRAIN_PTH = "D:/datasets/Low-light image dataset/train/low/*"
# TEST_PATH = "D:/datasets/Low-light image dataset/test/*"

TRAIN_PTH = "D:/datasets/Low-light image dataset/train/low/*"
TEST_PATH = "D:/datasets/Low-light image dataset/test/test_flame/*/*"
#--------------------------------------------------------------------------
IMAGE_SIZE = 256
ZOOM_SIZE = 300
LR_SIZE = 64
batch_size = 8
num_workers = 2
#-----------------------------------------------------------------
weak_tfms = transforms.Compose([
    transforms.Resize((ZOOM_SIZE,ZOOM_SIZE)),
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
tfms = transforms.Compose([       # Tensor
    # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])   
])
low_tfms = transforms.Compose([       # Tensor
    transforms.Resize((LR_SIZE, LR_SIZE)),
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])   
])
edge_tfms = transforms.Compose([
    # transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Resize the image
    transforms.Grayscale(),  # Convert image to grayscale
    transforms.Lambda(lambda x: x.filter(ImageFilter.FIND_EDGES)),  # Find edges
    transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the tensor
])
tfms_test = transforms.Compose([       # Tensor
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(), 
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])   
])
def get_image_files(folder_path):
    # Define the allowed image file extensions
    allowed_extensions = ["*.jpg", "*.jpeg", "*.png", "*.gif", "*.bmp"]  # Add more if needed

    image_files = []
    for ext in allowed_extensions:
        # Use glob to find files with allowed extensions in the folder path
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))

    return image_files
#-----------------------------------------------------------------
def get_highpath(f):
    original_path = Path(f)
    parts = list(original_path.parts)
    parts[parts.index('low')] = 'high'
    return Path(*parts)

def RandomSameCrop(img1, img2, scale=(0.8, 1.0)):
    size=(IMAGE_SIZE, IMAGE_SIZE)
    i, j, h, w = transforms.RandomResizedCrop.get_params(img1, scale=scale, ratio=(1.0, 1.0))
    # Apply the same crop to both images
    img1 = transforms.functional.resized_crop(img1, i, j, h, w, size)
    img2 = transforms.functional.resized_crop(img2, i, j, h, w, size)

    return img1, img2
def apply_transforms(img0, img1):

    img0 = TF.resize(img0, (IMAGE_SIZE, IMAGE_SIZE))
    img1 = TF.resize(img1, (IMAGE_SIZE, IMAGE_SIZE))
    
    if random.random() > 0.5:
        img0 = TF.vflip(img0)
        img1 = TF.vflip(img1)
    
    if random.random() > 0.5:
        img0 = TF.hflip(img0)
        img1 = TF.hflip(img1)
        
    # if random.random() > 0.3:
    #     e =  random.uniform(0.0,25.0)
    #     img0 = TF.rotate(img0, angle=e, expand=True)
    #     img1 = TF.rotate(img1,  angle=e, expand=True)
        
        
    return img0, img1

def add_gaussian_noise_to_images(x, mean=0, std=0.01):
    if random.random() < 0.5:
        noise = torch.randn_like(x) * std + mean
        x = x + noise
    return x