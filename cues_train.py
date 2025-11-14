import sys
sys.path.append('models')
sys.path.append('data')

import config
from data.mydataset import train_loader_cue

from utility import save_Cue_model, init_Cue
from loss import BCE, L1, Perceptual
from tqdm import tqdm
import torch

def train(model, opt, scr, loader, bce, criterion, lp):
    loop = tqdm(loader, leave=True)
    for idx, (x, y, LR, ED) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        LR, ED = LR.to(config.DEVICE), ED.to(config.DEVICE)

        with torch.cuda.amp.autocast():   
            LR_, ED_ = model(x)        
            edge_loss = criterion(ED_, ED)
            LRR_loss = criterion(LR_, LR) + lp(LR_, LR)
            loss = edge_loss + LRR_loss

        opt.zero_grad()
        scr.scale(loss).backward()
        scr.step(opt)
        scr.update()

        loop.set_postfix(
            loss=loss.item(),
        )
    return model, opt, scr

def main():
    trn_loader = train_loader_cue()
    #-----------------------------------------------------------------
    model, opt, scr = init_Cue()
    #-----------------------------------------------------------------
    bce = BCE(config.DEVICE)
    criterion = L1(config.DEVICE)
    Lp = Perceptual(config.DEVICE)
    #-----------------------------------------------------------------
    for epoch in range(config.NUM_EPOCHS):
        model, opt, scr = train(model, opt, scr, trn_loader, bce, criterion, Lp)
        save_Cue_model(model, opt)

if __name__ == "__main__":
    main()
