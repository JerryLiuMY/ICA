import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def plot_callback(history):
    """ Plot training and validation history
    :param history: training and validation history
    :return:
    """

    train_history, valid_history = history

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(train_history, label="train_loss")
    ax.plot(valid_history, label="valid_loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("ELBO")
    ax.legend(loc="upper right")

    return fig
