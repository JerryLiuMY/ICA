from data_prep.generator import generate_data
from data_prep.loader import load_data
from mle.simulation import simu_mle
from tools.params import exp_dict
from vae.simulation import simu_vae


def simulation(m, n, activation, model_name, outputs, seed, dist, scale):
    """ run simulation and reconstruction
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param activation: activation function for mlp
    :param model_name: name of the model to run
    :param outputs: outputs from the training function
    :param seed: random seed for dgp
    :param dist: distribution function for generating z
    :param scale: scale of the distribution for generating z
    """

    # define simulation functions
    simu_size = exp_dict["simu_size"]
    simu_dict = {"vae": simu_vae, "mleauto": simu_mle, "mlesgd": simu_mle}
    simu_func = simu_dict[model_name]

    # run simulation and reconstruction
    simu_df = generate_data(m, n, activation, simu_size, seed=seed, dist=dist, scale=scale)
    simu_loader = load_data(simu_df)
    recon_df = simu_func(outputs, simu_loader)

    return simu_df, recon_df
