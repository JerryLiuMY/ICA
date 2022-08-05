from data.generator import generate_data
from global_settings import DATA_PATH
import matplotlib.pyplot as plt
from torch import nn
import seaborn as sns
import pandas as pd
import os
sns.set()


def visualize():
    """ Visualize latent variables z and the generated variables x
    :return:
    """

    # initialize figure
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(6, 10)

    # first line
    m, n = 1, 1
    ax00 = fig.add_subplot(gs[0:2, 0:2])
    ax01 = fig.add_subplot(gs[0:2, 2:4])
    ax02 = fig.add_subplot(gs[0:2, 4:6])
    ax03 = fig.add_subplot(gs[0:2, 6:8])
    ax04 = fig.add_subplot(gs[0:2, 8:10])
    params_1_1 = pd.read_pickle(os.path.join(DATA_PATH, f"params_{m}_{n}.pkl"))
    data_df01 = generate_data(m=m, n=n, params=params_1_1, activation=nn.ReLU(), size=1000)
    data_df02 = generate_data(m=m, n=n, params=params_1_1, activation=nn.Sigmoid(), size=1000)
    data_df03 = generate_data(m=m, n=n, params=params_1_1, activation=nn.Tanh(), size=1000)
    data_df04 = generate_data(m=m, n=n, params=params_1_1, activation=nn.GELU(), size=1000)
    sns.histplot(data_df01, x="z0", kde=True, ax=ax00)
    sns.histplot(data_df01, x="x0", kde=True, ax=ax01)
    sns.histplot(data_df02, x="x0", kde=True, ax=ax02)
    sns.histplot(data_df03, x="x0", kde=True, ax=ax03)
    sns.histplot(data_df04, x="x0", kde=True, ax=ax04)
    ax00.set_title("dist. of z")
    ax01.set_title("dist. of x with ReLU")
    ax02.set_title("dist. of x with Sigmoid")
    ax03.set_title("dist. of x with Tanh")
    ax04.set_title("dist. of x with GELU")

    # second line
    m, n = 1, 2
    ax11 = fig.add_subplot(gs[2:4, 2:4])
    ax12 = fig.add_subplot(gs[2:4, 4:6])
    ax13 = fig.add_subplot(gs[2:4, 6:8])
    ax14 = fig.add_subplot(gs[2:4, 8:10])
    params_1_2 = pd.read_pickle(os.path.join(DATA_PATH, f"params_{m}_{n}.pkl"))
    data_df11 = generate_data(m=m, n=n, params=params_1_2, activation=nn.ReLU(), size=1000)
    data_df12 = generate_data(m=m, n=n, params=params_1_2, activation=nn.Sigmoid(), size=1000)
    data_df13 = generate_data(m=m, n=n, params=params_1_2, activation=nn.Tanh(), size=1000)
    data_df14 = generate_data(m=m, n=n, params=params_1_2, activation=nn.GELU(), size=1000)
    sns.kdeplot(data=data_df11, x="x0", y="x1", fill=True, ax=ax11)
    sns.kdeplot(data=data_df12, x="x0", y="x1", fill=True, ax=ax12)
    sns.kdeplot(data=data_df13, x="x0", y="x1", fill=True, ax=ax13)
    sns.kdeplot(data=data_df14, x="x0", y="x1", fill=True, ax=ax14)

    # third line
    m, n = 2, 2
    ax20 = fig.add_subplot(gs[4:6, 0:2])
    ax21 = fig.add_subplot(gs[4:6, 2:4])
    ax22 = fig.add_subplot(gs[4:6, 4:6])
    ax23 = fig.add_subplot(gs[4:6, 6:8])
    ax24 = fig.add_subplot(gs[4:6, 8:10])
    params_2_2 = pd.read_pickle(os.path.join(DATA_PATH, f"params_{m}_{n}.pkl"))
    data_df11 = generate_data(m=m, n=n, params=params_2_2, activation=nn.ReLU(), size=1000)
    data_df12 = generate_data(m=m, n=n, params=params_2_2, activation=nn.Sigmoid(), size=1000)
    data_df13 = generate_data(m=m, n=n, params=params_2_2, activation=nn.Tanh(), size=1000)
    data_df14 = generate_data(m=m, n=n, params=params_2_2, activation=nn.GELU(), size=1000)
    sns.kdeplot(data=data_df11, x="z0", y="z1", fill=True, ax=ax20)
    sns.kdeplot(data=data_df11, x="x0", y="x1", fill=True, ax=ax21)
    sns.kdeplot(data=data_df12, x="x0", y="x1", fill=True, ax=ax22)
    sns.kdeplot(data=data_df13, x="x0", y="x1", fill=True, ax=ax23)
    sns.kdeplot(data=data_df14, x="x0", y="x1", fill=True, ax=ax24)

    # add label
    fig.text(-0.01, 0.19, "m=2, n=2", va="center", rotation="vertical")
    fig.text(-0.01, 0.52, "m=1, n=2", va="center", rotation="vertical")
    fig.text(-0.01, 0.84, "m=1, n=1", va="center", rotation="vertical")
    fig.tight_layout()
    fig.savefig(os.path.join(DATA_PATH, "data_dist.pdf"), bbox_inches="tight")
