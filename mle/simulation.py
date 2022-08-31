from global_settings import device
import pandas as pd
import torch


def simu_mle(model, simu_loader):
    """ Perform simulation for reconstruction with MLE
    :param model: trained model for performing simulation
    :param simu_loader: simulation dataset loader
    :return: dataframe of reconstructions
    """

    # load parameters and initialize
    mean, logs2 = torch.empty(size=(0, model.n)), torch.empty(size=(0, 1))

    # perform simulation
    model.eval()
    nbatch = 0
    for _, z_batch in simu_loader:
        with torch.no_grad():
            z_batch = z_batch.to(device)
            mean_batch, logs2_batch = model(z_batch)
            mean, logs2 = torch.cat([mean, mean_batch], dim=0), torch.cat([logs2, logs2_batch], dim=0)
            nbatch += 1

    # simulation dataframe
    mean, logs2 = mean.cpu().detach().numpy(), logs2.cpu().detach().numpy()
    mean_dict = {f"mean{i}": mean[:, i].reshape(-1) for i in range(mean.shape[1])}
    logs2_dict = {f"logs2": logs2[:, 0].reshape(-1)}
    recon_dict = {**mean_dict, **logs2_dict}
    recon_df = pd.DataFrame(recon_dict)

    return recon_df
