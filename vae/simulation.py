from data_prep.generator import generate_data
from data_prep.loader import load_data


def simu_vae(model, simu_loader):
    """ Perform simulation with for non-linear ICA
    :param model: trained model for performing simulation
    :param simu_loader: simulation dataset loader
    :return: dataframe of reconstructions
    """

    # perform simulation
    model.eval()

    # save simulated data and reconstruction

    return recon_df
