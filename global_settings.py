from pathlib import Path
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODE_FLAG = "experiment"


def get_dir(path):
    """ Make directory and return path
    :param path:
    :return:
    """

    if not os.path.isdir(path):
        os.mkdir(path)

    return path


# define directories
DESKTOP_PATH = str(Path(os.getcwd()).parent.absolute())
if DESKTOP_PATH == "/Users/mingyu/Desktop":
    DRIVE_PATH = "/Volumes/Sumsung_1T/ICA"
    DATA_PATH = get_dir(os.path.join(DRIVE_PATH, "data_prep"))
    VAE_PATH = get_dir(os.path.join(DRIVE_PATH, "vae"))
    MLEAUTO_PATH = get_dir(os.path.join(DRIVE_PATH, "mleauto"))
    MLESGD_PATH = get_dir(os.path.join(DRIVE_PATH, "mlesgd"))
else:
    DATA_PATH = get_dir(os.path.join(DESKTOP_PATH, "data_prep"))
    VAE_PATH = get_dir(os.path.join(DESKTOP_PATH, "vae"))
    MLEAUTO_PATH = get_dir(os.path.join(DESKTOP_PATH, "mleauto"))
    MLESGD_PATH = get_dir(os.path.join(DESKTOP_PATH, "mlesgd"))

PATH_DICT = {"vae": VAE_PATH, "mleauto": MLEAUTO_PATH, "mlesgd": MLESGD_PATH}
