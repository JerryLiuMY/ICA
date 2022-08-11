from params.params import batch_size
from datetime import datetime
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import torch


def load_data(data_df):
    """ Load data from the specified dataset as data loader
    :param data_df: dataframe of z and x
    :return: dataset loader and input shape
    """

    # get x and z values for train dataset
    x_train_cols = [col for col in data_df.columns if "x" in col]
    z_train_cols = [col for col in data_df.columns if "z" in col]
    x_train = torch.tensor(data_df[x_train_cols].values.astype(np.float32))
    z_train = torch.tensor(data_df[z_train_cols].values.astype(np.float32))
    train_data = TensorDataset(x_train, z_train)

    # build dataset
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    input_size = train_data[0][0].shape[0]
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Loaded data with input_shape={input_size}")

    return train_loader
