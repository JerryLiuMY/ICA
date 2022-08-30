from vae.model import VAE, elbo_gaussian
from params.params import vae_dict as train_dict
from global_settings import device
from datetime import datetime
import numpy as np
import torch


def train_vae(m, n, train_loader, valid_loader, llh_func):
    """ Training VAE with the specified image dataset
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param train_loader: training dataset loader
    :param valid_loader: validation dataset loader
    :param llh_func: function for numerical integration
    :return: trained model and training loss history
    """

    # load parameters
    epochs, lr, beta = train_dict["epochs"], train_dict["lr"], train_dict["beta"]

    # building VAE
    model = VAE(m, n)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.995)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Number of parameters: {num_params}")

    # training loop
    model.train()
    train_loss_li, train_llh_li = [], []
    valid_loss_li, valid_llh_li = [], []

    for epoch in range(epochs):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training on epoch {epoch} "
              f"[lr={round(scheduler.get_last_lr()[0], 6)}]...")

        # mini-batch loop
        train_loss, train_llh, nbatch = 0., 0., 0
        for x_batch, z_batch in train_loader:
            x_batch, z_batch = x_batch.to(device), z_batch.to(device)
            mean_batch, logs2_batch, mu_batch, logvar_batch = model(x_batch)
            loss = elbo_gaussian(x_batch, mean_batch, logs2_batch, mu_batch, logvar_batch, beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() / x_batch.size(dim=0)
            train_llh += llh_func(m, n, x_batch, z_batch, model) / x_batch.size(dim=0)
            nbatch += 1

        # get training loss and llh
        scheduler.step()
        train_loss = train_loss / nbatch
        train_llh = train_llh / nbatch
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Finish epoch {epoch} "
              f"with loss={round(train_loss, 2)} and llh={round(train_llh, 2)}")
        train_loss_li.append(train_loss)
        train_llh_li.append(train_llh)

        # get validation loss
        valid_loss, valid_llh = valid_vae(m, n, model, valid_loader, llh_func, eval_mode=False)
        valid_loss_li.append(valid_loss)
        valid_llh_li.append(valid_llh)

    # return train/valid history and log-likelihoods
    train_loss_arr = np.array(train_loss_li)
    valid_loss_arr = np.array(valid_loss_li)
    train_llh_arr = np.array(train_llh_li)
    valid_llh_arr = np.array(valid_llh_li)
    callback = {
        "loss": [train_loss_arr, valid_loss_arr],
        "llh": [train_llh_arr, valid_llh_arr]
    }

    return model, callback


def valid_vae(m, n, model, valid_loader, llh_func, eval_mode):
    """ Training VAE with the specified image dataset
    :param m: dimension of the latent variable
    :param n: dimension of the target variable
    :param model: trained VAE model
    :param valid_loader: validation dataset loader
    :param llh_func: function for numerical integration
    :param eval_mode: whether set to evaluation model
    :return: validation loss
    """

    # load parameters and set evaluation mode
    beta = train_dict["beta"]
    if eval_mode:
        model.eval()

    # get validation loss
    valid_loss, valid_llh, nbatch = 0., 0., 0
    for x_batch, z_batch in valid_loader:
        with torch.no_grad():
            x_batch, z_batch = x_batch.to(device), z_batch.to(device)
            mean_batch, logs2_batch, mu_batch, logvar_batch = model(x_batch)
            loss = elbo_gaussian(x_batch, mean_batch, logs2_batch, mu_batch, logvar_batch, beta)

            valid_loss += loss.item() / x_batch.size(dim=0)
            valid_llh += llh_func(m, n, x_batch, z_batch, model) / x_batch.size(dim=0)
            nbatch += 1

    valid_loss = valid_loss / nbatch
    valid_llh = valid_llh / nbatch
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Finish validation "
          f"with loss={round(valid_loss, 2)} and llh={round(valid_llh, 2)}")

    return valid_loss, valid_llh
