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

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(train_loss, label="train_loss")
    ax.plot(valid_loss, label="valid_loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("ELBO")
    ax.legend(loc="upper right")

    return fig
