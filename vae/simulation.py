from global_settings import DEVICE
import pandas as pd
import torch


def simu_vae(inputs, simu_loader):
    """ Perform simulation for reconstruction with VAE
    :param inputs: trained model for performing simulation
    :param simu_loader: simulation dataset loader
    :return: dataframe of reconstructions
    """

    # load parameters and initialize
    [model] = inputs
    x_recon, logs2 = torch.empty(size=(0, model.n)), torch.empty(size=(0, 1))
    mu, logvar = torch.empty(size=(0, model.m)), torch.empty(size=(0, model.m))

    # perform simulation
    model.eval()
    nbatch = 0
    for x_batch, _ in simu_loader:
        with torch.no_grad():
            x_batch = x_batch.to(DEVICE)
            x_recon_batch, logs2_batch, mu_batch, logvar_batch = model(x_batch)
            x_recon, logs2 = torch.cat([x_recon, x_recon_batch], dim=0), torch.cat([logs2, logs2_batch], dim=0)
            mu, logvar = torch.cat([mu, mu_batch], dim=0), torch.cat([logvar, logvar_batch], dim=0)
            nbatch += 1

    # simulation dataframe
    x_recon, logs2 = x_recon.cpu().detach().numpy(), logs2.cpu().detach().numpy()
    mu, logvar = mu.cpu().detach().numpy(), logvar.cpu().detach().numpy()
    x_recon_dict = {f"x_recon{i}": x_recon[:, i].reshape(-1) for i in range(x_recon.shape[1])}
    logs2_dict = {f"logs2": logs2[:, 0].reshape(-1)}
    mu_dict = {f"mu{i}": mu[:, i].reshape(-1) for i in range(mu.shape[1])}
    logvar_dict = {f"logvar{i}": logvar[:, i].reshape(-1) for i in range(logvar.shape[1])}
    recon_dict = {**x_recon_dict, **logs2_dict, **mu_dict, **logvar_dict}
    recon_df = pd.DataFrame(recon_dict)

    return recon_df
