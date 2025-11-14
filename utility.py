import torch
import config
import torch.optim as optim
# from models.msff import MSFF
from models.generator import Generator
from models.discriminator import Discriminator
from models.cue import Cue_Net
from torchvision.utils import save_image

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def save_model(model, disc, opt, opt_disc):
    if config.SAVE_checkpoints:
        save_checkpoint(model, opt, filename=config.GEN_checkpoints)
        save_checkpoint(disc, opt_disc, filename=config.DISC_checkpoints)

def save_Cue_model(model, opt):
    if config.SAVE_checkpoints:
        save_checkpoint(model, opt, filename=config.CUE_checkpoints)

def save_some_examples(model, cue, cbdef, loader, epoch, folder):
    
    x = next(iter(loader))
    x = x.to(config.DEVICE)

    model.eval()
    with torch.no_grad():
        LR, ED = cue(x)
        _, _, _, C = cbdef(x)
        fake = model(x, LR, ED, C)
        

        # concat_cover = torch.cat((x*.5+.5, y*.5+.5, fake*.5+.5), 2)
        concat_cover = torch.cat((x*.5+.5, fake*.5+.5), 2)
        save_image(concat_cover, folder + f"gen_{epoch}.png")
      
    model.train()

def init_Cue():
    cue = Cue_Net(in_channels=3, features=64).to(config.DEVICE)
    opt = optim.Adam(cue.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    scr = torch.cuda.amp.GradScaler()
    if config.LOAD_checkpoints:
        load_checkpoint(config.CUE_checkpoints, cue, opt, config.LEARNING_RATE)
    return cue, opt, scr

def init_Generator():
    model = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    scr = torch.cuda.amp.GradScaler()
    if config.LOAD_checkpoints:
        load_checkpoint(config.GEN_checkpoints, model, opt, config.LEARNING_RATE)
    return model, opt, scr

def init_DISC():
    disc = Discriminator().to(config.DEVICE)
    opt = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    scr = torch.cuda.amp.GradScaler()
    if config.LOAD_checkpoints:
        load_checkpoint(config.DISC_checkpoints, disc, opt, config.LEARNING_RATE)
    return disc, opt, scr

def train_disc(disc, opt, scr, x, fake, real, bce):
    
    with torch.cuda.amp.autocast():
        D_real = disc(x, real)
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake = disc(x, fake.detach())
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

    opt.zero_grad()
    scr.scale(D_loss).backward()
    scr.step(opt)
    scr.update() 
    
    return disc, opt, scr, D_loss

def train_model(gen, disc, opt, scr, x, fake, y, bce, criterion, lp):
    
    with torch.cuda.amp.autocast():
        D_fake = disc(x, fake)
        adv = bce(D_fake, torch.ones_like(D_fake))

        total = criterion(y, fake)*config.L1_LAMBDA + lp(y, fake) + adv

    opt.zero_grad()
    scr.scale(total).backward()
    scr.step(opt)
    scr.update()

    return gen, opt, scr, total
