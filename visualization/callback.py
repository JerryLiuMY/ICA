import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def plot_callback(history, llh):
    """ Plot training and validation history
    :param history: training and validation history
    :param llh: training and validation log-likelihood
    :return:
    """

    train_history, valid_history = history
    train_llh, valid_llh = llh

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(train_history, label="train_loss")
    ax.plot(valid_history, label="valid_loss")
    ax.plot(train_llh, label="train_llh")
    ax.plot(valid_llh, label="valid_llh")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_ylabel("Log-likelihood")

    return fig
