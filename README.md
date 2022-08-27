# MLE/VAE for nonlinear ICA
<p>
    <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-v3-brightgreen.svg" alt="python"></a> &nbsp;
</p>

The up-to-date <a href="./__resources__/ICA/main.pdf" target="_blank">write-up</a> and the <a href="https://www.overleaf.com/project/62e45e862465cfc8d3bc6aed" target="_blank">overleaf</a> project.

Dictionary of parameters: https://github.com/JerryLiuMY/ICA/blob/main/params/params.py

## Data Information
<a href="https://drive.google.com/drive/folders/1OnsuFWZwtcZhROKImRHxXBBkdrAlD5Ti?usp=sharing" target="_blank">Repository</a> for the generated `weight` and `bias` of the single-layer MLP.


```python
from torch import nn
from data_prep.generator import generate_data
from data_prep.loader import load_data
from params.params import exp_dict

m, n, activation = 10, 20, nn.ReLU()

# training data
train_size = exp_dict["train_size"]
train_df = generate_data(m, n, activation, train_size)
train_loader = load_data(train_df)

# validation data
valid_size = exp_dict["valid_size"]
valid_df = generate_data(m, n, activation, valid_size)
valid_loader = load_data(valid_df)

# simulation data
simu_size = exp_dict["simu_size"]
simu_df = generate_data(m, n, activation, simu_size)
simu_loader = load_data(simu_df)
```

Sample distribution of the latent variable `z` and the generated `x`
- **Column**: Distribution of `z`, `x` with `ReLU` activation, `x` with `Sigmoid` activation, `x` with `Tanh` activation and `x` with `GELU` activation
- **Row**: Distribution of `z` and `x` with `m=1, n=1`, `m=1, n=2` and `m=2, n=2`

![alt text](./__resources__/data_dist.jpg?raw=true "Title")

## Variational Method
<a href="https://drive.google.com/drive/folders/1OpN3lfy2Eew5eH-7AY1A6-2v6GMcxcq1?usp=sharing" target="_blank">Folder</a> for the trained VAE models. <a href="./vae">Link</a> to the model architecture, training loop and simulation (<a href="https://github.com/JerryLiuMY/ICA/blob/8adb6fcbe68ba727bb4856913fe99bbad84640f7/vae/vae.py#L92">ELBO</a> with Gaussian MLP as decoder).

```python
from vae.training import train_vae
from vae.training import valid_vae
from vae.simulation import simu_vae

model, train_loss = train_vae(m, n, train_loader, valid_loader)
valid_loss = valid_vae(m, n, valid_loader, model, eval_mode=True)
recon_df = simu_vae(m, n, model, simu_loader)
```

### Setting with `m=2, n=20`
Setting with `m=2, n=20, sigma^2=1` for different types of activation functions `ReLU`, `Sigmoid`, `Tanh` and `GELU`.

- #### Observation and Reconstruction
![alt text](./__resources__/recon_m2_n20.jpg?raw=true "Title")

- #### Prior and Posterior
![alt text](./__resources__/latent_m2_n20.jpg?raw=true "Title")

- #### Learning Curve
![alt text](./__resources__/callback_m2_n20_mc.jpg?raw=true "Title")


### Setting with `m=2, n=2`
Setting with `m=2, n=2, sigma^2=1` for different types of activation functions `ReLU`, `Sigmoid`, `Tanh` and `GELU`.

- #### Observation and Reconstruction
![alt text](./__resources__/recon_m2_n2.jpg?raw=true "Title")

- #### Prior and Posterior
![alt text](./__resources__/latent_m2_n2.jpg?raw=true "Title")

- #### Learning Curve
![alt text](./__resources__/callback_m2_n2_mc.jpg?raw=true "Title")


## MLE with Gradient Descent
<a href="./funcs/likelihood.py">Link</a> to numerical integration for the likelihood function. 
