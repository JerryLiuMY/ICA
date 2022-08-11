from params.params import batch_size
from datetime import datetime
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch


def load_data(train_df, valid_df):
    """ Load data from the specified dataset as data loader
    :param train_df: training dataframe of z and x
    :param valid_df: validation dataframe of z and x
    :return: dataset loader and input shape
    """

    # get x and z values for train dataset
    x_train_cols = [col for col in train_df.columns if "x" in col]
    z_train_cols = [col for col in train_df.columns if "z" in col]
    x_train = torch.tensor(train_df[x_train_cols].values)
    z_train = torch.tensor(train_df[z_train_cols].values)
    train_data = TensorDataset(x_train, z_train)

    # get x and z values for valid dataset
    x_valid_cols = [col for col in valid_df.columns if "x" in col]
    z_valid_cols = [col for col in valid_df.columns if "z" in col]
    x_valid = torch.tensor(valid_df[x_valid_cols].values)
    z_valid = torch.tensor(valid_df[z_valid_cols].values)
    valid_data = TensorDataset(x_valid, z_valid)

    # build dataset
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    input_size = train_data[0][0].shape[0]
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Loaded data with input_shape={input_size}")

    return train_loader, valid_loader
