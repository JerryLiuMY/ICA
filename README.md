# MLE/VAE for nonlinear ICA
<p>
    <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-v3-brightgreen.svg" alt="python"></a> &nbsp;
</p>

<a href="./__resources__/ICA/main.pdf" target="_blank">Link</a> to the updated write-up of the project.  

Dictionary of parameters: https://github.com/JerryLiuMY/VAE/blob/main/params/params.py

```python
from data.generator import generate_data
from data.loader import load_data
from vae.train import train_vae
from vae.train import valid_vae
from torch import nn

# load data and perform training & validation
m, n, activation, train_size, valid_size = 10, 20, nn.ReLU(), 10000, 2000
train_df, valid_df = generate_data(m, n, activation, train_size, valid_size)
train_loader, valid_loader = load_data(train_df, valid_df)
model, train_loss = train_vae(train_loader)
valid_loss = valid_vae(model, valid_loader)
```

## Data Information
<a href="https://drive.google.com/drive/folders/1Uep9CpOhQor72GXVWeb7ax2kO7O7wFco?usp=sharing" target="_blank">Repository</a> for the generated `weight` and `bias` of the single-layer MLP. Distribution of the latent variable `z` and the generated `x`
- **Column**: Distribution of `z`, `x` with `ReLU` activation, `x` with `Sigmoid` activation, `x` with `Tanh` activation and `x` with `GELU` activation
- **Row**: Distribution of `z` and `x` with `m=1, n=1`, `m=1, n=2` and `m=2, n=2`

![alt text](./__resources__/data_dist.jpg?raw=true "Title")

## VAE
<a href="https://drive.google.com/drive/folders/1HNsTgwhNfs60Dx9ef7eQuOsU6ftaono8?usp=sharing">Folder</a> for the trained VAE model and visualizations.
