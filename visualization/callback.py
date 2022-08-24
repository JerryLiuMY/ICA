import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def plot_callback(loss, llh):
    """ Plot training and validation history
    :param loss: training and validation loss
    :param llh: training and validation llh
    :return:
    """

    train_loss, valid_loss = loss
    train_llh, valid_llh = llh

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(train_loss, label="train_loss")
    ax.plot(valid_loss, label="valid_loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("ELBO")
    ax.legend(loc="upper left")

    ax_ = ax.twinx()
    ax_.plot(train_llh, label="train_llh")
    ax_.plot(valid_llh, label="valid_llh")
    ax_.set_ylabel("Log-likelihood")
    ax_.legend(loc="upper right")

    return fig
