from pathlib import Path
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define directories
DESKTOP_PATH = str(Path(os.getcwd()).parent.absolute())
if DESKTOP_PATH == "/Users/mingyu/Desktop":
    DRIVE_PATH = "/Volumes/Sumsung_1T/ICA"
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


# Find likelihood -- a) sparse grid b) monte carlo
# Find gradient -- a) SGD b) MLE gradient descent
# Compare generated images with reconstruction
# Compare latent variable p(z|x) with estimates
# Compare latent variable p(z) with estimates
# Modify modification to derived gradient of likelihood


# Underestimation of variance
# Think about identification requirements
