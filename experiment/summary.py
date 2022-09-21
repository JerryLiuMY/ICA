from visualization.callback import plot_callback
from visualization.recon import plot_recon_2d
from matplotlib import pyplot as plt
import json
import os


def summary(m, n, model_name, log_path, exp_path, llh_method):
    """ Plot original space, latent space and callback
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param model_name: model name
    :param exp_path: path for logging
    :param log_path: path for experiment
    :param llh_method: method for numerical integration
    """

    # plot recon and callback
    callback, metrics = plot_callback(m, n, model_name, exp_path, llh_method=llh_method)
    callback.savefig(os.path.join(log_path, f"callback_m{m}_n{n}_{llh_method}.pdf"), bbox_inches="tight")
    with open(os.path.join(log_path, f"metrics.json"), "w") as handle:
        json.dump(metrics, handle)
    recon = plot_recon_2d(m, n, exp_path)
    recon.savefig(os.path.join(log_path, f"recon_m{m}_n{n}.pdf"), bbox_inches="tight")
    plt.close(callback)
    plt.close(recon)
