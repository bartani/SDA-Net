import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from thop import profile

class Cue_Net(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(Cue_Net, self).__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 1, 1, padding_mode="reflect"),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Conv2d(features, features*2, 3, 1, 1, padding_mode="reflect"),
            nn.BatchNorm2d(features*2),
            nn.ReLU()
        )
        
        self.ED_path = nn.Sequential(
            nn.Conv2d((features*2)+3, features*4, 3, 1, 1, padding_mode="reflect"),
            nn.BatchNorm2d(features*4),
            nn.ReLU(),
            nn.Conv2d(features*4, 1, 3, 1, 1, padding_mode="reflect"),
            nn.Tanh()
        )
        self.LR_path = nn.Sequential(
            nn.Conv2d((features*2)+3, features*4, 4, 2, 1, padding_mode="reflect"),
            nn.BatchNorm2d(features*4),
            nn.ReLU(),
            nn.Conv2d(features*4, 3, 4, 2, 1, padding_mode="reflect"),
            nn.Tanh()
        )

    def forward(self, x):
        x_ = self.initial_down(x)

        LR = self.LR_path(torch.cat([x , x_], 1))
        ED = self.ED_path(torch.cat([x , x_], 1))
        return LR, ED 

def init_Cue(DEVICE, LEARNING_RATE, MODEL_checkpoints):
    model = Cue_Net(in_channels=3, features=64).to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999),)
    scr = torch.cuda.amp.GradScaler()
    load_checkpoint(MODEL_checkpoints, model, opt, LEARNING_RATE, DEVICE)
    return model, opt, scr

def load_checkpoint(checkpoint_file, model, optimizer, lr, DEVICE):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def test():
    DEVICE = "cuda"
    LEARNING_RATE = 1e-4
    MODEL_checkpoints = "D:/Papers Code/Low-Light Image Enhancement/Cue-Net/weights/LOL/cue.pth.tar"
    x = torch.randn((8, 3, 256, 256)).to(DEVICE)
    model, _, _ = init_Cue(DEVICE, LEARNING_RATE, MODEL_checkpoints)
    # model = encoder(in_channels=3, features=64)
    LR, ED = model(x)
    print("Cue_Net output shape:", LR.shape, ED.shape)

# Example usage
if __name__ == "__main__":
    test()
    # batch_size = 8
    # x = torch.randn(batch_size, 3, 256, 256)  # Input tensor
    # gud = Cue_Net(in_channels=3, features=64)
    # LR, ED  = gud(x)
    # print("Cue_Net output shape:", LR.shape, ED.shape)
    # LR, ED = torch.randn(batch_size, 3, 64, 64), torch.randn(batch_size, 1, 256, 256)

    # flops, params = profile(model, inputs=(x, LR, ED, C1, C2, C3))
    # print("flops:", flops/1e9)
    # print("params:", params/1e6)