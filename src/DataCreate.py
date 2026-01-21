# -*- coding: utf-8 -*-
import numpy as np
import random
import matplotlib as plt
from sklearn.preprocessing import normalize
#### MODEL PARAMETERS

X_variance = 1
noise_variance = 1
thetavalue = 10

# Data creation: X, Y, theta, sparsity indices

def CreateData(n: int = 100, p1: int = 50, sparsity: int = 5, noise: float = 1):

    X = normalize(np.random.normal(0, X_variance, size = (n, p1)), axis=0, norm='max')
    sparsity_indices = random.sample(range(1, p1), sparsity)
    theta = np.array([0 if i not in sparsity_indices else thetavalue for i in range(p1)]).reshape(p1,1)
    Y = X@theta + noise * np.random.normal(0, noise_variance, size=(n, 1))

    return X, Y, theta, sorted(sparsity_indices)

