import sys
sys.path.append('models')
sys.path.append('data')

import config
from data.mydataset import train_loader_CBDFE
from utility import init_Encode, save_CBDFE_model
from tqdm import tqdm
import torch
from loss import SimCLRLoss

def train(model, opt, scr, loader, criterion, epoch):
    loop = tqdm(loader, leave=True)
    for idx, (x, _, x_pos, x_neg) in enumerate(loop):
        x = x.to(config.DEVICE)
        x_pos, x_neg = x_pos.to(config.DEVICE), x_neg.to(config.DEVICE)

        #.view(y.size(0), -1)
        with torch.cuda.amp.autocast():
            a = model(x)
            p = model(x_pos)
            loss = criterion(a.view(a.size(0), -1), p.view(p.size(0), -1))

        opt.zero_grad()
        scr.scale(loss).backward()
        scr.step(opt)
        scr.update()


        loop.set_postfix(
            LOSS=loss.item(),
            epoch=epoch,
        )
    return model, opt, scr

def main():
    trn_loader = train_loader_CBDFE()
    #-----------------------------------------------------------------
    model, opt, scr = init_Encode()
    #-----------------------------------------------------------------
    criterion = SimCLRLoss()
    #-----------------------------------------------------------------
    for epoch in range(config.NUM_EPOCHS):
        model, opt, scr = train( model, opt, scr, trn_loader, criterion, epoch)        
        save_CBDFE_model(model, opt)

if __name__ == "__main__":
    main()

