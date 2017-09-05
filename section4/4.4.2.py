from mpl_toolkits.mplot3d import Axes3D
import sys, os
from common import lib as l
from common import mnist as m
import pickle

class simpleNet:
    def __init__(self):
        self.W = l.np.random.randn(2,3)

    def predict(self, x):
        return l.np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = l.softmax(z)
        loss = l.cross_entropy_error(y, t)

        return loss

net = simpleNet()
#print(net.W)

x = l.np.array([0.6, 0.9])
t = l.np.array([0, 0, 1])

def f(W):
    return net.loss(x, t)

dW = l.numerical_gradient(f, net.W)
print(dW)
