from vae.vae import VariationalAutoencoder, elbo_gaussian
from funcs.likelihood import get_llh_mc
from params.params import vae_dict as train_dict
from global_settings import device
from datetime import datetime
import numpy as np
import torch


def train_vae(m, n, train_loader, valid_loader):
    """ Training VAE with the specified image dataset
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param train_loader: training dataset loader
    :param valid_loader: validation dataset loader
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
    train_loss_li, train_llh_li = [], []
    valid_loss_li, valid_llh_li = [], []
    for epoch in range(epoch):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training on epoch {epoch} "
              f"[lr={round(scheduler.get_last_lr()[0], 6)}]...")

        # training and get training loss
        train_loss, train_llh, nbatch = 0., 0., 0
        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            mean_batch, logs2_batch, mu_batch, logvar_batch = model(x_batch)
            input_batch = [x_batch, logs2_batch]
            loss = elbo_gaussian(x_batch, mean_batch, logs2_batch, mu_batch, logvar_batch, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() / x_batch.size(dim=0)
            train_llh += get_llh_mc(m, n, input_batch, model) / x_batch.size(dim=0)
            nbatch += 1

        scheduler.step()
        train_loss = train_loss / nbatch
        train_llh = train_llh / nbatch
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Finish epoch {epoch} with loss {train_loss}")
        train_loss_li.append(train_loss)
        train_llh_li.append(train_llh)

        # validation and get validation loss
        valid_loss, valid_llh = valid_vae(m, n, valid_loader, model, eval_mode=False)
        valid_loss_li.append(valid_loss)
        valid_llh_li.append(valid_llh)

    # return train/valid history and log-likelihoods
    train_loss_arr = np.array(train_loss_li)
    valid_loss_arr = np.array(valid_loss_li)
    loss = [train_loss_arr, valid_loss_arr]

    return model, loss


def valid_vae(m, n, valid_loader, model, eval_mode):
    """ Training VAE with the specified image dataset
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param model: trained VAE model
    :param valid_loader: validation dataset loader
    :param eval_mode: whether set to evaluation model
    :return: validation loss
    """

    # load parameters and set evaluation mode
    beta = train_dict["beta"]
    if eval_mode:
        model.eval()

    # get validation loss
    valid_loss, valid_llh, nbatch = 0., 0., 0
    for x_batch, _ in valid_loader:
        with torch.no_grad():
            x_batch = x_batch.to(device)
            mean_batch, logs2_batch, mu_batch, logvar_batch = model(x_batch)
            input_batch = [x_batch, logs2_batch]
            loss = elbo_gaussian(x_batch, mean_batch, logs2_batch, mu_batch, logvar_batch, beta)

            valid_loss += loss.item() / x_batch.size(dim=0)
            valid_llh += get_llh_mc(m, n, input_batch, model) / x_batch.size(dim=0)
            nbatch += 1

    valid_loss = valid_loss / nbatch
    valid_llh = valid_llh / nbatch
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Finish validation with loss {valid_loss}")

    return valid_loss, valid_llh
