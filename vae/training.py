from vae.vae import VariationalAutoencoder, elbo_gaussian
from params.params import vae_dict as train_dict
from global_settings import device
from datetime import datetime
import torch
import numpy as np


def train_vae(m, n, train_loader):
    """ Training VAE with the specified image dataset
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param train_loader: training dataset loader
    :return: trained model and training loss history
    """

    # load parameters
    epoch, lr, beta = train_dict["epoch"], train_dict["lr"], train_dict["beta"]

    # building VAE
    model = VariationalAutoencoder(m, n)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.995)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Number of parameters: {num_params}")

    # training loop
    model.train()
    train_loss = []
    for epoch in range(epoch):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training on epoch {epoch} "
              f"[lr={round(scheduler.get_last_lr()[0], 6)}]...")

        epoch_loss, nbatch = 0., 0
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            mean_batch, logs2_batch, mu_batch, logvar_batch = model(x_batch)
            loss = elbo_gaussian(x_batch, mean_batch, logs2_batch, mu_batch, logvar_batch, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() / x_batch.size(dim=0)
            nbatch += 1

        scheduler.step()

        # append training loss
        epoch_loss = epoch_loss / nbatch
        train_loss.append(epoch_loss)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Finish epoch {epoch} with loss {epoch_loss}")

    train_loss = np.array(train_loss)

    return model, train_loss


def valid_vae(model, valid_loader):
    """ Training VAE with the specified image dataset
    :param model: trained VAE model
    :param valid_loader: validation dataset loader
    :return: validation loss
    """

    # load parameters
    beta = train_dict["beta"]

    # set to evaluation mode
    model.eval()
    valid_loss, nbatch = 0., 0
    for x_batch, _ in valid_loader:
        with torch.no_grad():
            x_batch = x_batch.to(device)
            mean_batch, logs2_batch, mu_batch, logvar_batch = model(x_batch)
            loss = elbo_gaussian(x_batch, mean_batch, logs2_batch, mu_batch, logvar_batch, beta)
            valid_loss += loss.item() / x_batch.size(dim=0)
            nbatch += 1

    # report validation loss
    valid_loss = valid_loss / nbatch
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Finish validation with loss {valid_loss}")

    return valid_loss
