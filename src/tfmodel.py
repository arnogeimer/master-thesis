import numpy as np
import tensorflow as tf
import tqdm
from matplotlib import pyplot as plt

def extracttheta(w1):
    p1 = w1.shape[0]
    thetahat = np.zeros(p1)
    for i in range(p1):
        thetahat[i] = (w1[i,2*i] - w1[i,2*i+1])/2
    return thetahat

class Tfmodel(object):
    def __init__(self, Xsample, Ysample, activation, lamb, learningRate, thetavalue, sparsity_indices, oracle, ifTrain=True):
        self.Xsample = Xsample
        self.Ysample = Ysample
        self.activation = activation
        self.lamb = lamb
        self.learningRate = learningRate
        self.thetavalue = thetavalue
        self.sparsity_indices = sparsity_indices
        self.oracle = oracle
        self.loss = []

        n, p1 = self.Xsample.shape

        self.Ysample.reshape(n,1)
        b1_ini = np.random.normal(loc=0., scale=1, size=(1, 2*p1))
        w2_ini = np.random.normal(loc=0., scale=0.05, size=(2*p1,1))
        b2_ini = np.random.normal(loc=0., scale=1, size=(1,1))
        if not self.oracle:
            # if not Oracle, we shall regenerate the parameters randomly
            w1_ini = np.random.uniform(0., 10, size=(p1,2*p1))
        else:
            w1_ini = np.zeros((p1,2*p1))
            for i in self.sparsity_indices:
                w1_ini[i, 2*i] = self.thetavalue
                w1_ini[i, 2*i+1] = -self.thetavalue

        self.w1 = tf.Variable(w1_ini ,shape=(p1,2*p1), trainable=True)
        self.b1 = tf.Variable(b1_ini ,shape=(1,2*p1), trainable=True)
        self.w2 = tf.Variable(w2_ini ,shape=(2*p1,1), trainable=True)
        self.b2 = tf.Variable(b2_ini ,shape=(1,1), trainable=True)

    @tf.function
    def forward(self):

        # new structure in TF2.0 to track the parts of computation that will need backprop
        with tf.GradientTape(persistent=True) as g:

            # these are the parameters of which we will need the gradients
            g.watch([self.w1,self.b1,self.w2,self.b2])

            # the structure is used muptiple times, we should just compute once X \cdot w_1
            Xw1 = tf.matmul(self.Xsample, self.w1)

            w2_L2Norm = tf.sqrt(tf.reduce_sum(self.w2*self.w2, axis=0))

            yhat = self.b2 + tf.matmul(self.activation(Xw1+self.b1), self.w2/w2_L2Norm)

            bareCost = tf.sqrt(tf.reduce_sum(tf.square(yhat-self.Ysample), axis=0)) # sqrt Lasso
            #bareCost = tf.reduce_sum(tf.square(yhat-self.Ysample), axis=0)
            cost = bareCost + self.lamb*tf.reduce_sum(tf.abs(self.w1))

        # computing cost gradients
        w1_gra = g.gradient(cost, self.w1)
        b1_gra = g.gradient(cost, self.b1)
        w2_gra = g.gradient(cost, self.w2)
        b2_gra = g.gradient(cost, self.b2)

        del g

        self.w1.assign_sub(self.learningRate*w1_gra)
        # update b1 with constraints
        # first we compute the bounding envolope

        maxXw1 = tf.reduce_max(Xw1, axis=0)
        minXw1 = tf.reduce_min(Xw1, axis=0)

        # proposition
        proposed_b1 = self.b1 - self.learningRate*b1_gra

        # Booleans that determines if outside the bounds
        ifBelow = (proposed_b1 < minXw1)
        ifAbove = (proposed_b1 > maxXw1)

        # If outside, take the bounds
        proposed_b1 = tf.where( ifBelow, minXw1, proposed_b1)
        proposed_b1 = tf.where( ifAbove, maxXw1, proposed_b1)

        #update b1, w2, b2
        self.b1.assign(proposed_b1)
        self.w2.assign_sub(self.learningRate*w2_gra)
        self.b2.assign_sub(self.learningRate*b2_gra)

        # return a list of gradients
        #return cost, [w1_gra,b1_gra,w2_gra,b2_gra]
        return cost

    def train(self, iterations: int = 100):
        for _ in tqdm.trange(iterations):      # steepest descent part
            cost = self.forward()
            self.loss.append(cost.numpy()[0])

        # we return only the numpy values
        thetahat = extracttheta(self.w1.numpy())
        return self.loss, thetahat

