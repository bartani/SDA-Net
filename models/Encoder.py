import torch
import torch.nn as nn
import torch.optim as optim


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


class encoder(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(encoder, self).__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, 1, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features*2, 3, 1, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features*2, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            features * 2, features * 2, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU(),
            nn.Sigmoid()
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        y = self.bottleneck(d7)
        return d1, d2, d3, y

def init_CBDFE(DEVICE, LEARNING_RATE, MODEL_checkpoints):
    model = encoder(in_channels=3, features=64).to(DEVICE)
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
    MODEL_checkpoints = "D:/Papers Code/Low-Light Image Enhancement/CBDFE/weights/LOL/enc_SimCLR.pth.tar"
    x = torch.randn((8, 3, 256, 256)).to(DEVICE)
    model, _, _ = init_CBDFE(DEVICE, LEARNING_RATE, MODEL_checkpoints)
    # model = encoder(in_channels=3, features=64)
    C1, C2, C3, y = model(x)
    print(y.shape)
    print(C1.shape, C2.shape, C3.shape)


if __name__ == "__main__":
    test()
