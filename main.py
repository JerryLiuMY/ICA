from experiment.experiment import experiment
from experiment.simulation import simulation
from experiment.summary import summary
from multiprocessing import Pool
from functools import partial
from torch import nn
import re
import os


def run_experiments(iter_m_n, model_name, exp_path, train_s2, decoder_dgp, llh_method="mc"):
    """ Perform experiments for non-linear ICA
    :param iter_m_n: iterable iter_m_n of experiments
    :param model_name: name of the model to run
    :param model_name: name of the model to run
    :param exp_path: path for experiment
    :param train_s2: whether to train s2 or not
    :param decoder_dgp: whether to use the same decoder as dgp
    :param llh_method: method for numerical integration
    """

    # define seed and llh_method
    seed = int(exp_path.split("/")[-1].split("_")[-1])
    llh_method = "null" if model_name == "vae" else llh_method

    # perform experiments
    for m, n in iter_m_n:
        run_experiment_multi(m, n, model_name=model_name, exp_path=exp_path,
                             train_s2=train_s2, decoder_dgp=decoder_dgp, llh_method=llh_method, seed=seed)


def run_experiment_multi(m, n, model_name, exp_path, train_s2, decoder_dgp, llh_method="mc", seed=0):
    """ Perform experiments for non-linear ICA with multiprocessing
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param model_name: name of the model to run
    :param exp_path: path for experiment
    :param train_s2: whether to train s2 or not
    :param decoder_dgp: whether to use the same decoder as dgp
    :param llh_method: method for numerical integration
    :param seed: random seed for dgp
    """

    # define list of activation functions
    activation_li = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU()]

    # perform experiment with multiprocessing
    experiment_func = partial(run_experiment, model_name=model_name, exp_path=exp_path,
                              train_s2=train_s2, decoder_dgp=decoder_dgp, llh_method=llh_method, seed=seed)
    iterable = [(m, n, activation) for activation in activation_li]
    pool = Pool(processes=len(iterable))
    pool.starmap(experiment_func, iterable=iterable)
    pool.close()
    pool.join()

    run_summary(m, n, model_name=model_name, exp_path=exp_path, llh_method=llh_method)


def run_experiment(m, n, activation, model_name, exp_path, train_s2, decoder_dgp, llh_method="mc", seed=0):
    """ Perform experiments for non-linear ICA
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param activation: activation function for mlp
    :param model_name: name of the model to run
    :param exp_path: path for experiment
    :param train_s2: whether to train s2 or not
    :param decoder_dgp: whether to use the same decoder as dgp
    :param llh_method: method for numerical integration
    :param seed: random seed for dgp
    """

    # define model path
    activation_name = ''.join([_ for _ in re.sub("[\(\[].*?[\)\]]", "", str(activation)) if _.isalpha()])
    model_path = os.path.join(exp_path, f"m{m}_n{n}_{activation_name}")

    # perform experiment and simulation
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
        outputs = experiment(m, n, activation, model_name=model_name, model_path=model_path,
                             train_s2=train_s2, decoder_dgp=decoder_dgp, llh_method=llh_method, seed=seed)
        simu_df, recon_df = simulation(m, n, activation, model_name=model_name, outputs=outputs,
                                       seed=seed, dist="normal", scale=1)
        simu_df.to_csv(os.path.join(model_path, "simu_df.csv"))
        recon_df.to_csv(os.path.join(model_path, "recon_df.csv"))


def run_summary(m, n, model_name, exp_path, llh_method="mc"):
    """ Perform experiments for non-linear ICA
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param model_name: name of the model to run
    :param exp_path: path for experiment
    :param llh_method: method for numerical integration
    """

    log_path = os.path.join(exp_path, f"m{m}_n{n}_log")
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
        summary(m, n, model_name=model_name, log_path=log_path, exp_path=exp_path, llh_method=llh_method)
