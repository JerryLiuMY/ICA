from pathlib import Path
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# define OUTPUT_PATH
DESKTOP_PATH = str(Path(os.getcwd()).parent.absolute())
if DESKTOP_PATH == "/Users/mingyu/Desktop":
    OUTPUT_PATH = "/Volumes/Sumsung_1T/ICA"
else:
    OUTPUT_PATH = os.path.join(DESKTOP_PATH, "ICA")

# define DATA_PATH
DATA_PATH = os.path.join(OUTPUT_PATH, "data_prep")
if not os.path.isdir(DATA_PATH):
    os.mkdir(DATA_PATH)

# define MODEL_PATH
VAE_PATH = os.path.join(OUTPUT_PATH, "vae")
MLEAUTO_PATH = os.path.join(OUTPUT_PATH, "mleauto")
MLESGD_PATH = os.path.join(OUTPUT_PATH, "mlesgd")
VAE_EXP_PATH = os.path.join(OUTPUT_PATH, "vae_exp")
MLEAUTO_EXP_PATH = os.path.join(OUTPUT_PATH, "mleauto_exp")
MLESGD_EXP_PATH = os.path.join(OUTPUT_PATH, "mlesgd_exp")
PATH_DICT = {
    "vae": VAE_PATH, "mleauto": MLEAUTO_PATH, "mlesgd": MLESGD_PATH,
    "vae_exp": VAE_EXP_PATH, "mleauto_exp": MLEAUTO_EXP_PATH, "mlesgd_exp": MLESGD_EXP_PATH
}


def get_dir(path):
    """ Make directory and return path
    :param path:
    :return: path
    """

    if not os.path.isdir(path):
        os.mkdir(path)

    return path


# plot for disparity
# investigate metrics related to CCA
