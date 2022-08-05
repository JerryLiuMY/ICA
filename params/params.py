import numpy as np
from torch import nn
np.random.seed(10)
n = 1
m = 1

# define coefficients
sigma = 3
w = np.random.rand(n, m)
b = np.random.rand(n)
coeffs = {"sigma": sigma, "w": w, "b": b}

# define parameters
params = {"activation": nn.Sigmoid()}  # ["ReLU", "Sigmoid", "Tanh"]
