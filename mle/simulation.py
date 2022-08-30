from global_settings import device
from datetime import datetime
import pandas as pd
import torch


def simu_mle(m, n, model, simu_loader):
    """ Perform simulation for reconstruction with VAE
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param model: trained model for performing simulation
    :param simu_loader: simulation dataset loader
    :return: dataframe of reconstructions
    """

    # load parameters and initialize
    mean, logs2 = torch.empty(size=(0, n)), torch.empty(size=(0, 1))

    # perform simulation
    model.eval()
    valid_loss, nbatch = 0., 0
    for x_batch, _ in simu_loader:
        with torch.no_grad():
            x_batch = x_batch.to(device)
            mean_batch, logs2_batch = model(x_batch)
            mean, logs2 = torch.cat([mean, mean_batch], dim=0), torch.cat([logs2, logs2_batch], dim=0)
            nbatch += 1

    # simulation dataframe
    mean, logs2 = mean.cpu().detach().numpy(), logs2.cpu().detach().numpy()
    mean_dict = {f"mean{i}": mean[:, i].reshape(-1) for i in range(mean.shape[1])}
    logs2_dict = {f"logs2": logs2[:, 0].reshape(-1)}
    recon_dict = {**mean_dict, **logs2_dict}
    recon_df = pd.DataFrame(recon_dict)

    # report simulation loss
    valid_loss = valid_loss / nbatch
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Finish validation with loss {valid_loss}")

    return recon_df
