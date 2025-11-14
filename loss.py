import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import clip
import torchvision.transforms as transforms

from math import exp
from math import pi

# from .common import VGG19, gaussian_blur

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class VGG19(nn.Module):
    def __init__(self, resize_input=False):
        super(VGG19, self).__init__()
        features = models.vgg19(weights="VGG19_Weights.IMAGENET1K_V1").features

        self.resize_input = resize_input
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        prefix = [1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5]
        posfix = [1, 2, 1, 2, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]
        names = list(zip(prefix, posfix))
        self.relus = []
        for pre, pos in names:
            self.relus.append('relu{}_{}'.format(pre, pos))
            self.__setattr__('relu{}_{}'.format(
                pre, pos), torch.nn.Sequential())

        nums = [[0, 1], [2, 3], [4, 5, 6], [7, 8],
                [9, 10, 11], [12, 13], [14, 15], [16, 17],
                [18, 19, 20], [21, 22], [23, 24], [25, 26],
                [27, 28, 29], [30, 31], [32, 33], [34, 35]]

        for i, layer in enumerate(self.relus):
            for num in nums[i]:
                self.__getattr__(layer).add_module(str(num), features[num])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # resize and normalize input for pretrained vgg19
        x = (x + 1.0) / 2.0
        x = (x - self.mean.view(1, 3, 1, 1)) / (self.std.view(1, 3, 1, 1))
        if self.resize_input:
            x = F.interpolate(
                x, size=(256, 256), mode='bilinear', align_corners=True)
        features = []
        for layer in self.relus:
            x = self.__getattr__(layer)(x)
            features.append(x)
        out = {key: value for (key, value) in list(zip(self.relus, features))}
        return out

class L1(): 
    def __init__(self,device):
        self.calc = torch.nn.L1Loss().to(device)
    
    def __call__(self, x, y):
        return self.calc(x, y)

class BCE(): 
    def __init__(self,device):
        self.calc = torch.nn.BCEWithLogitsLoss().to(device)
    
    def __call__(self, x, y):
        return self.calc(x, y)

class MSE(): 
    def __init__(self,device):
        self.calc = torch.nn.MSELoss().to(device)
    
    def __call__(self, x, y):
        return self.calc(x, y)

class Perceptual():
    def __init__(self, device, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        self.vgg = VGG19().to(device)
        self.criterion = torch.nn.L1Loss().to(device)
        self.weights = weights

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        prefix = [1, 2, 3, 4, 5]
        for i in range(5):
            content_loss += self.weights[i] * self.criterion(
                x_vgg[f'relu{prefix[i]}_1'], y_vgg[f'relu{prefix[i]}_1'])
        return content_loss

class Style(nn.Module):
    def __init__(self, device):
        super(Style, self).__init__()
        self.vgg = VGG19().to(device)
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * c)
        return G

    def __call__(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        style_loss = 0.0
        prefix = [2, 3, 4, 5]
        posfix = [2, 4, 4, 2]
        for pre, pos in list(zip(prefix, posfix)):
            style_loss += self.criterion(
                self.compute_gram(x_vgg[f'relu{pre}_{pos}']), self.compute_gram(y_vgg[f'relu{pre}_{pos}']))
        return style_loss
    
class TextualConsistencyLoss():
    def __init__(self, device='cuda'):
        self.device = device
        self.model, self.preprocess = clip.load('ViT-B/32', device=self.device)
        self.to_pil = transforms.ToPILImage()

    def __call__(self, image, text):
        
        loss = 0
        batch_size = image.shape[0]
        for b in range(batch_size):
            img = self.to_pil(image[b,:])
            image_input = self.preprocess(img).unsqueeze(0).to(self.device)
            # text_input = clip.tokenize(text[b]).to(self.device)
            
            image_features = self.model.encode_image(image_input)
            # text_features = self.model.encode_text(text_input)
            text_features = text[b,:]
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = torch.cosine_similarity(image_features, text_features)
            
            loss+=(1-similarity)
        
        return loss/batch_size

if __name__ == "__main__":
    pass