from params.params import batch_size
from datetime import datetime
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import torch


def load_data(data_df):
    """ Load data from the specified dataset as data loader
    :param data_df: dataframe of z and x
    :return: dataset loader for z and x
    """

    # get x and z values for train dataset
    x_cols = [col for col in data_df.columns if "x" in col]
    z_cols = [col for col in data_df.columns if "z" in col]
    x = torch.tensor(data_df[x_cols].values.astype(np.float32))
    z = torch.tensor(data_df[z_cols].values.astype(np.float32))
    dataset = TensorDataset(x, z)

    # build dataloader (not shuffled and reusable)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    input_size = dataset[0][0].shape[0]
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Loaded data with input_size={input_size}")

    return data_loader
