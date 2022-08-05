import numpy as np
from torch import nn
n = 1
m = 1

# define coefficients
sigma = 3
np.random.seed(10)
w = np.random.rand(n, m)
b = np.random.rand(n)
coeffs = {"sigma": sigma, "w": w, "b": b}

# define parameters
params = {"activation": nn.Tanh()}  # ["ReLU", "Sigmoid", "Tanh"]
