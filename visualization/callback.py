import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def plot_callback(history):
    """ Plot training and validation history
    :param history: training and validation history
    :return:
    """

    train_history, valid_history = history

    fig, ax = plt.subplots(1, 1, figsize=(3, 6))
    ax.plot(train_history, label="train_loss")
    ax.plot(valid_history, label="valid_loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_ylabel("Log-likelihood")

    return fig
