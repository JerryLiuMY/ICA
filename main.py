from experiment.experiment import experiment
from experiment.summary import summarize
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

    seed = int(exp_path.split("/")[-1].split("_")[-1])
    llh_method = "null" if model_name == "vae" else llh_method
    for m, n in iter_m_n:
        run_experiment_multi(m, n, model_name=model_name, exp_path=exp_path,
                             train_s2=train_s2, decoder_dgp=decoder_dgp, llh_method=llh_method, seed=seed)


def run_experiment_multi(m, n, model_name, exp_path, train_s2, decoder_dgp, llh_method="mc", seed=0):
    """ Perform experiments for non-linear ICA with multi-processing
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param model_name: name of the model to run
    :param exp_path: path for experiment
    :param train_s2: whether to train s2 or not
    :param decoder_dgp: whether to use the same decoder as dgp
    :param llh_method: method for numerical integration
    :param seed: random seed for dgp
    """

    activation_li = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU()]
    for activation in activation_li:
        run_experiment(m, n, activation, model_name=model_name, exp_path=exp_path,
                       train_s2=train_s2, decoder_dgp=decoder_dgp, llh_method=llh_method, seed=seed)
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

    activation_name = ''.join([_ for _ in re.sub("[\(\[].*?[\)\]]", "", str(activation)) if _.isalpha()])
    model_path = os.path.join(exp_path, f"m{m}_n{n}_{activation_name}")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
        experiment(m, n, activation, model_name=model_name, model_path=model_path,
                   train_s2=train_s2, decoder_dgp=decoder_dgp, llh_method=llh_method, seed=seed)


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
        summarize(m, n, model_name=model_name, log_path=log_path, exp_path=exp_path, llh_method=llh_method)
