import sys
sys.path.append('models')
sys.path.append('data')

import config
from data.mydataset import train_loader, test_loader
from models.Encoder import init_CBDFE
from models.cue import init_Cue
from utility import init_Generator, init_DISC, save_model, save_some_examples, train_disc, train_model
from loss import BCE, MSE, L1, Perceptual
from tqdm import tqdm
import torch

def train(
    model, opt, scr,
    disc, opt_disc, scr_disc,
    cue, cbdfe, loader, bce, criterion, lp 
):
    loop = tqdm(loader, leave=True)
    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)

        with torch.no_grad():
            LR, ED = cue(x)
            _, _, _, C = cbdfe(x)
        
        fake = model(x, LR, ED, C)

        disc, opt_disc, scr_disc, D_loss = train_disc(disc, opt_disc, scr_disc, x, fake, y, bce)

        model, opt, scr, total = train_model(model, disc, opt, scr, x, fake, y, bce, criterion, lp)

        loop.set_postfix(
            # CUE=cue_loss.item(),
            # MODEL = fake.shape,
            DISC=D_loss.item(),
            # CBDFE=cbdfe_loss.item(),
            MODEL=total.item(),
        )
    return model, opt, scr, disc, opt_disc, scr_disc

def main():
    #-------------------------------datasets----------------------------------
    trn_loader = train_loader()
    tst_loader = test_loader()
    #-------------------------------pre trained models------------------------
    cbdfe, _, _ = init_CBDFE(config.DEVICE, config.LEARNING_RATE, config.CBDFE_checkpoints)
    cue, _, _ = init_Cue(config.DEVICE, config.LEARNING_RATE, config.CUE_checkpoints)
    #-------------------------------init main models--------------------------
    model, opt, scr = init_Generator()
    disc, opt_disc, scr_disc = init_DISC()
    #-------------------------------initil saving-----------------------------
    # save_model(model, disc, opt, opt_disc)
    # save_some_examples(model, cue, cbdfe, tst_loader, 10000, f"outcomes/enhanced/")
    #-----------------------------------------------------------------
    bce = BCE(config.DEVICE)
    criterion = L1(config.DEVICE)
    Lp = Perceptual(config.DEVICE)
    # # Ls = Style(config.DEVICE)
    
    # #-----------------------------------------------------------------
    for epoch in range(config.NUM_EPOCHS):
        model, opt, scr, disc, opt_disc, scr_disc = train(
            model, opt, scr, disc, opt_disc, scr_disc, cue, cbdfe, trn_loader, bce, criterion, Lp
        )        
        save_model(model, disc, opt, opt_disc)
        save_some_examples(model, cue, cbdfe, tst_loader, epoch, f"outcomes/enhanced/")

if __name__ == "__main__":
    main()


