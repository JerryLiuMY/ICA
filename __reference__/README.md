# MLE/VAE for nonlinear ICA [Notes]

## Numerical Integration
### Monte Carlo
- Computationally efficient
- Work well in high dimension (large `m`)
- High variance and require large enough samples

### Sparse Grid
- Computationally less efficient
- Numerical underflow in high dimension 
- Can evaluate exactly with fine enough grid + complete coverage

## Training and Inference
### Learning curve
- **VAE:** The ELBO converges reasonably fast
- **VAE:** The LLH first deteriorates before improving again
- **VAE:** The LLH can deteriorate as we train for more iterations (e.g. Sigmoid)
- **MLE:** The LLH converges very fast
- **MLE:** Autograd does not work for `n=20` need to switch down to `n=10`

### Sigma^{2}
- **VAE:** `sigma^{2}` estimated using network with VAE and can be data dependent
- **VAE:** `sigma^{2}` under estimated with VAE 
- **MLE:** `sigma^{2}` estimated using an independent variable
- **MLE:** `sigma^{2}` estimated correctly estimated with MLE 

## Observed Space
- Compare `p(x|z)` with PCA
- Single latent sample per observation
- `m=2, n=20` has good reconstruction property (in particular for `ReLU` and `GELU`)
- `m=2, n=2` may experience over-parametrization
- **VAE:** `p(x|z)` is generated from the variational distribution
- **MLE:** `p(x|z)` is generated from the prior

## Latent Space
- MCMC for computing the posterior `p(z|x)`
- The posterior `p(z|x)` resembles the observed space more than the latent
- Single latent sample per observation
- **VAE:** Latent `\hat{p}(z|x)` available for VAE 
- **VAE:** Latent variable `z` of interest or by-product of VAE structure
- **MLE:** Latent `\hat{p}(z|x)` not available for MLE
