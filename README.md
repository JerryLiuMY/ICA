# MLE/VAE for nonlinear ICA
<p>
    <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-v3-brightgreen.svg" alt="python"></a> &nbsp;
</p>

<a href="./__resources__/ICA/main.pdf" target="_blank">Link</a> to the updated write-up of the project.

Dictionary of parameters: https://github.com/JerryLiuMY/ICA/blob/main/params/params.py

## Data Information
<a href="https://drive.google.com/drive/folders/1Uep9CpOhQor72GXVWeb7ax2kO7O7wFco?usp=sharing" target="_blank">Repository</a> for the generated `weight` and `bias` of the single-layer MLP.

```python
from torch import nn
from data_prep.generator import generate_data
from data_prep.loader import load_data
from params.params import exp_dict

m, n, activation = 10, 20, nn.ReLU()
train_size = exp_dict["train_size"]
valid_size = exp_dict["valid_size"]
simu_size = exp_dict["simu_size"]
train_df = generate_data(m, n, activation, train_size)
valid_df = generate_data(m, n, activation, valid_size)
simu_df = generate_data(m, n, activation, simu_size)
train_loader = load_data(train_df)
valid_loader = load_data(valid_df)
simu_loader = load_data(simu_df)
```

Distribution of the latent variable `z` and the generated `x`
- **Column**: Distribution of `z`, `x` with `ReLU` activation, `x` with `Sigmoid` activation, `x` with `Tanh` activation and `x` with `GELU` activation
- **Row**: Distribution of `z` and `x` with `m=1, n=1`, `m=1, n=2` and `m=2, n=2`

![alt text](./__resources__/data_dist.jpg?raw=true "Title")

## VAE
<a href="https://drive.google.com/drive/folders/1HNsTgwhNfs60Dx9ef7eQuOsU6ftaono8?usp=sharing">Folder</a> for the trained VAE models. <a href="./vae/vae.py">Link</a> to the model architecture and <a href="./vae/training.py">link</a> to the training loop (<a href="https://github.com/JerryLiuMY/ICA/blob/2c7748a05fac87e021c789e7202e89d3814cb2a1/vae/training.py#L90">ELBO</a> with Gaussian MLP as decoder).

```python
from vae.training import train_vae
from vae.training import valid_vae
from vae.simulation import simu_vae

model, train_loss = train_vae(m, n, train_loader)
valid_loss = valid_vae(model, valid_loader)
recon_df = simu_vae(m, n, model, simu_loader)
```

Distribution of the original and reconstructed latent variables `z` with `m=2, n=20` for different types of activation functions `ReLU`, `Sigmoid`, `Tanh` and `GELU`

![alt text](./__resources__/vae_latent.jpg?raw=true "Title")
