from global_settings import device
import pandas as pd
import numpy as np
import torch


def simu_mle(inputs, simu_loader):
    """ Perform simulation for reconstruction with MLE
    :param inputs: trained VAE model and log of the fitted s2
    :param simu_loader: simulation dataset loader
    :return: dataframe of reconstructions
    """

    # load parameters and initialize
    [model, logs2] = inputs
    mean = torch.empty(size=(0, model.n))

    # perform simulation
    model.eval()
    nbatch = 0
    for _, z_batch in simu_loader:
        with torch.no_grad():
            z_batch = z_batch.to(device)
            mean_batch = model(z_batch)
            mean = torch.cat([mean, mean_batch], dim=0)
            nbatch += 1

    # simulation dataframe
    mean, logs2 = mean.cpu().detach().numpy(), logs2.cpu().detach().numpy()
    mean_dict = {f"mean{i}": mean[:, i].reshape(-1) for i in range(mean.shape[1])}
    logs2_dict = {f"logs2": np.repeat(logs2, mean.shape[0])}
    recon_dict = {**mean_dict, **logs2_dict}
    recon_df = pd.DataFrame(recon_dict)

    return recon_df
