import sys
sys.path.append('models')
sys.path.append('data')

import config
from data.mydataset import test_loader
from utility import init_Generator, init_Cue, init_Encode
from tqdm import tqdm
import torch
from torchvision.utils import save_image

def main():
    #-------------------------------datasets----------------------------------
    loader = test_loader()
    #-------------------------------pre trained models------------------------
    cbdfe, _, _ = init_Encode()
    cue, _, _ = init_Cue()
    #-------------------------------init main models--------------------------
    model, _, _ = init_Generator()
    #-------------------------------initil saving-----------------------------
    cbdfe.eval()
    cue.eval()
    model.eval()
    loop = tqdm(loader, leave=True)
    for idx, (x) in enumerate(loop):
        x = x.to(config.DEVICE)

        with torch.no_grad():
            LR, ED = cue(x)
            C = cbdfe(x)
        fake = model(x, LR, ED, C)
        b = fake.shape[0]
        for i in range(b):
            f = fake[i]
            concat_cover = torch.cat((x[i]*.5+.5, f*.5+.5), 1)
            save_image(concat_cover, f"outcomes/enhanced/gen_{i}_{idx}.png")

        # concat_cover = torch.cat((x*.5+.5, fake*.5+.5), 2)
        # save_image(fake*.5+.5, f"outcomes/NPE/gen_{idx}.png")
    

if __name__ == "__main__":
    main()


