import tqdm
import src.DataCreate as dc
import src.tfmodel as tfm
import src.evaluation as ev
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt

def draw_losses_3d(losses: List, xrange: List, yrange: List, figname: str = "./Figure3d.pdf"):
    fig, ax = plt.subplots()
    im = ax.imshow(losses, cmap='hot',  origin = 'lower')
    fig.colorbar(im, ax=ax)

    ax.set_xlabel('Non-zero entries')
    ax.set_ylabel('Feature size')

    plt.savefig(figname, bbox_inches="tight")

losses = []
srange = np.linspace(1, 25, 25, dtype=int)
yrange = np.linspace(10, 50, 40, dtype=int)

for y_size in tqdm.tqdm(yrange):
    losses_y = []
    for sparsity in srange:
        X, Y, theta, spars = dc.CreateData(5000, y_size + sparsity, sparsity, noise = 1)

        tfmodel = tfm.Tfmodel(X, Y, tf.nn.relu, 0.1, 0.01, 10, spars, oracle = True)
        _, thetahat = tfmodel.train(1000)
        losses_y.append(ev.MCC(theta, thetahat))
    losses.append(losses_y)

print(losses)

draw_losses_3d(losses, srange, yrange, figname="./Figure3d_tfmodel.pdf")