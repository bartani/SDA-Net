import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
import time

def add_gaussian_noise_to_latent(latent_space, mean=0, std=0.01):
    noise = torch.randn_like(latent_space) * std + mean
    noisy_latent_space = latent_space + noise
    return noisy_latent_space

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x

class DCB(nn.Module):
    def __init__(self, F_feature, C_Feature):
        super(DCB, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(F_feature+C_Feature, F_feature)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(C_Feature, F_feature)
        )

    def forward(self, F, C):
        C = C.view(C.size(0), -1)
        F = F.view(F.size(0), -1)
        z = self.layer1(torch.cat([F, C], 1))
        C = self.layer2(C)
        return z*C + F
    
class GLDC(nn.Module):
    def __init__(self, F_feature, C_Feature):
        super(GLDC, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C_Feature, C_Feature, 4, 2, 1, padding_mode="reflect"),
            nn.ReLU()
        )
        self.dcb1 = DCB(F_feature, C_Feature)
        self.dcb2 = DCB(F_feature, C_Feature)
        self.dcb3 = DCB(F_feature, C_Feature)
        self.dcb4 = DCB(F_feature, C_Feature)
        # self.dcb5 = DCB(F_feature, C_Feature)
        # self.dcb6 = DCB(F_feature, C_Feature)
        # self.dcb7 = DCB(F_feature, C_Feature)
        # self.dcb8 = DCB(F_feature, C_Feature)
        # self.dcb9 = DCB(F_feature, C_Feature)
        # self.dcb10 = DCB(F_feature, C_Feature)

    def forward(self, F, C):
        C = self.conv(C)
        F = self.dcb1(F, C)
        F = self.dcb2(F, C)
        F = self.dcb3(F, C)
        F = self.dcb4(F, C)
        # F = self.dcb5(F, C)
        # F = self.dcb6(F, C)
        # F = self.dcb7(F, C)
        # F = self.dcb8(F, C)
        # F = self.dcb9(F, C)
        # F = self.dcb10(F, C)
        F = F.unsqueeze(-1).unsqueeze(-1)
        return F

class Generator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.GLDC = GLDC(features*8, features*8)
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels*2+1, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def initializing_data(self, x, LR, ED):
        return torch.cat([x, ED, F.interpolate(LR, scale_factor=4, mode='bilinear', align_corners=False)], 1)

    def forward(self, x, LR, ED, C):
        x = self.initializing_data(x, LR, ED)
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6) #*
        F = self.bottleneck(d7)
        F = add_gaussian_noise_to_latent(F)
        F = self.GLDC(F, C)
        up1 = self.up1(F)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        return self.final_up(torch.cat([up7, d1], 1))


def test():
    batch_size = 1
    x = torch.randn((batch_size, 3, 256, 256)).to('cuda')
    LR, ED = torch.randn(batch_size, 3, 64, 64), torch.randn(batch_size, 1, 256, 256)
    y  = torch.randn(batch_size, 512, 2, 2)

    model = Generator(in_channels=3, features=64).to('cuda')

    model.eval()
    preds = model(x, LR, ED, y)
    print(preds.shape)

    
    flops, params = profile(model, inputs=(x, LR, ED, y))
    print("flops:", flops/1e9)
    print("params:", params/1e6)


def check_time():
    batch_size = 1
    x = torch.randn((batch_size, 3, 256, 256)).to('cuda')
    LR, ED = torch.randn(batch_size, 3, 64, 64).to('cuda'), torch.randn(batch_size, 1, 256, 256).to('cuda')
    y  = torch.randn(batch_size, 512, 2, 2).to('cuda')

    model = Generator(in_channels=3, features=64).to('cuda')

    model.eval()    

    num_runs = 1000
    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x, LR, ED, y)
    
    torch.cuda.synchronize()
    end = time.time()

    avg_time = (end-start)/num_runs
    fpg = 1/avg_time
    print(f"time:{avg_time*100:.2f} ms")

def Modelparameters():
    model = Generator(in_channels=3, features=64).to('cuda')
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model size: {num_params/1e6:.2f} M")

if __name__ == "__main__":
    Modelparameters()
