from data_prep.generator import generate_data
from data_prep.loader import load_data


def simu_vae(m, n, model, activation, simu_size):
    """ Perform simulation with for non-linear ICA
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param model: trained model for performing simulation
    :param activation: activation function for mlp
    :param simu_size: number of samples for simulation
    :return: dataframe of z and x
    """

    # load data for simulation
    data_df = generate_data(m, n, activation, simu_size)
    data_loader = load_data(data_df)

    # perform simulation
    model.eval()

    # save simulated data and reconstruction

    return recon_df
