from pathlib import Path
import torch
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# define directories
DESKTOP_PATH = str(Path(os.getcwd()).parent.absolute())
if DESKTOP_PATH == "/Users/mingyu/Desktop":
    DRIVE_PATH = "/Volumes/Sumsung_1T/Projects/ICA"
    DATA_PATH = os.path.join(DRIVE_PATH, "data_prep")
    MLE_PATH = os.path.join(DRIVE_PATH, "mle")
    VAE_PATH = os.path.join(DRIVE_PATH, "vae")
else:
    raise ValueError("Invalid path")

# make directories
if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)

if not os.path.isdir(MLE_PATH):
    os.mkdir(MLE_PATH)

if not os.path.isdir(VAE_PATH):
    os.mkdir(VAE_PATH)


# save reconstruction -- mu, logvar, mean, logs2
# plot latent space
# ICA formulation

# check derivation for MLE
