import src.DataCreate as dc
import src.tfmodel as tfm
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt

def draw_losses_3d(losses: List, figname: str = "./Figure3d.pdf"):
    fig, ax = plt.subplots()
    im = ax.imshow(losses, cmap='hot', interpolation='bilinear', origin = 'lower')
    fig.colorbar(im, ax=ax)

    plt.savefig(figname, bbox_inches="tight")

losses = []

for y_size in np.linspace(15, 100, 6, dtype=int):
    losses_y = []
    for sparsity in np.linspace(1, 10, 5, dtype=int):
        X, Y, theta, spars = dc.CreateData(10, y_size, sparsity, True)

        tfmodel = tfm.Tfmodel(X, Y, tf.nn.relu, 0.1, 0.01, 10, spars, True)
        _, thetahat = tfmodel.train(1000)
        thetahat = thetahat / np.linalg.norm(thetahat, ord=2)
        theta = theta / np.linalg.norm(theta, ord=2)
        losses_y.append(np.linalg.norm(thetahat - theta.reshape(-1), ord=2))
    losses.append(losses_y)

draw_losses_3d(losses, figname="./Figure3d_tfmodel.pdf")