from vae.vae import VariationalAutoencoder
from params.params import vae_dict as train_dict
from global_settings import device
from datetime import datetime
import torch
import numpy as np


def train_vae(m, n, train_loader):
    """ Training VAE with the specified image dataset
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param train_loader: training image dataset loader
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

        for train_batch, _ in train_loader:
            train_batch = train_batch.to(device)
            train_batch_mean, train_batch_logs2, mu, logvar = model(train_batch)
            loss = elbo_gaussian(train_batch, train_batch_mean, train_batch_logs2, mu, logvar, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update loss and nbatch
            epoch_loss += loss.item() / train_batch.size(dim=0)
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
    :param valid_loader: validation image dataset loader
    :return: validation loss
    """

    # load parameters
    beta = train_dict["beta"]

    # set to evaluation mode
    model.eval()
    valid_loss, nbatch = 0., 0.
    for valid_batch, _ in valid_loader:
        with torch.no_grad():
            valid_batch = valid_batch.to(device)
            valid_batch_mean, valid_batch_logs2, mu, logvar = model(valid_batch)
            loss = elbo_gaussian(valid_batch, valid_batch_mean, valid_batch_logs2, mu, logvar, beta)

            # update loss and nbatch
            valid_loss += loss.item() / valid_batch.size(dim=0)
            nbatch += 1

    # report validation loss
    valid_loss = valid_loss / nbatch
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Finish validation with loss {valid_loss}")

    return valid_loss


def elbo_gaussian(x, mean, logs2, mu, logvar, beta):
    """ Calculating loss for variational autoencoder
    :param x: original image
    :param mean: mean in the output layer
    :param logs2: log of the variance in the output layer
    :param mu: mean in the hidden layer
    :param logvar: log of the variance in the hidden layer
    :param beta: beta
    :return: reconstruction loss + KL
    """

    # KL-divergence
    kl_div = - 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # reconstruction loss
    recon_loss = - torch.sum(logs2.mul(x.size(dim=1)/2) + torch.norm(x - mean, 2, dim=1).pow(2).div(logs2.exp().mul(2)))

    # loss
    loss = - beta * kl_div + recon_loss

    return -loss
