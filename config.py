import torch



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#-----------------------------------------------------------------
LOAD_checkpoints = False
SAVE_checkpoints = True

CBDFE_checkpoints = f"weights/enc_SimCLR.pth.tar"
CUE_checkpoints = f"weights/cue.pth.tar"


GEN_checkpoints = f"weights/gen.pth.tar"
DISC_checkpoints = f"weights/disc.pth.tar"
#-----------------------------------------------------------------
LEARNING_RATE = 1e-4
NUM_EPOCHS = 200
L1_LAMBDA = 100
#-----------------------------------------------------------------
