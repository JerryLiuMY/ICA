from pathlib import Path
from utils.tools import get_dir
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define directories
DESKTOP_PATH = str(Path(os.getcwd()).parent.absolute())
if DESKTOP_PATH == "/Users/mingyu/Desktop":
    DRIVE_PATH = "/Volumes/Sumsung_1T/ICA"
    DATA_PATH = get_dir(os.path.join(DRIVE_PATH, "data_prep"))
    MLE_PATH = get_dir(os.path.join(DRIVE_PATH, "mle"))
    VAE_PATH = get_dir(os.path.join(DRIVE_PATH, "vae"))
else:
    DATA_PATH = get_dir(os.path.join(DESKTOP_PATH, "data_prep"))
    MLE_PATH = get_dir(os.path.join(DESKTOP_PATH, "mle"))
    VAE_PATH = get_dir(os.path.join(DESKTOP_PATH, "vae"))


# Compare sparse grid vs. monte carlo
# Find gradient -- a) autograd b) MLE gradient descent

# Underestimation of variance
# Think about identification requirements
