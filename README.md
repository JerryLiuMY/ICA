# MLE/VAE for nonlinear ICA
<p>
    <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-v3-brightgreen.svg" alt="python"></a> &nbsp;
</p>

<a href="./__resources__/ICA/main.pdf" target="_blank">Link</a> to the updated write-up of the project.

## Data Information
Dictionary of parameters: https://github.com/JerryLiuMY/ICA/blob/main/params/params.py

<a href="https://drive.google.com/drive/folders/1Uep9CpOhQor72GXVWeb7ax2kO7O7wFco?usp=sharing" target="_blank">Repository</a> for the generated `weight` and `bias` of the single-layer MLP.

```python
from torch import nn
from data.generator import generate_data
from data.loader import load_data

m, n, activation, train_size, valid_size = 10, 20, nn.ReLU(), 10000, 2000
train_df, valid_df = generate_data(m, n, activation, train_size, valid_size)
train_loader, valid_loader = load_data(train_df, valid_df)
```

Distribution of the latent variable `z` and the generated `x`
- **Column**: Distribution of `z`, `x` with `ReLU` activation, `x` with `Sigmoid` activation, `x` with `Tanh` activation and `x` with `GELU` activation
- **Row**: Distribution of `z` and `x` with `m=1, n=1`, `m=1, n=2` and `m=2, n=2`

![alt text](./__resources__/data_dist.jpg?raw=true "Title")

## VAE
<a href="https://drive.google.com/drive/folders/1HNsTgwhNfs60Dx9ef7eQuOsU6ftaono8?usp=sharing">Folder</a> for the trained VAE models. <a href="./models/vae.py">Link</a> to the model architecture and <a href="./experiments/train_vae.py">link</a> to the training loop (<a href="https://github.com/JerryLiuMY/ICA/blob/955ad3fc26c19cfb9b6da82a528254e3094cbca2/experiments/train_vae.py#L88">ELBO</a> with Gaussian MLP as decoder).

```python
from experiments.train_vae import train_vae
from experiments.train_vae import valid_vae

model, train_loss = train_vae(train_loader)
valid_loss = valid_vae(model, valid_loader)
```
