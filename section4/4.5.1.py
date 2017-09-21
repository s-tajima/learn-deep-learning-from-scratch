import sys, os
from common import lib as l

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -l.np.sum(t * l.np.log(y)) / batch_size

def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = l.np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 値を元に戻す
        
    return grad

def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = l.np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * l.np.random.randn(input_size, hidden_size)
        self.params['b1'] = l.np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * l.np.random.randn(hidden_size, output_size)
        self.params['b2'] = l.np.zeros(output_size)

    def sigmoid(self, x):
        return 1 / (1 + l.np.exp(-x))

    def softmax(self, a):
        c = l.np.max(a)
        exp_a = l.np.exp(a - c)
        sum_exp_a = l.np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = l.np.dot(x, W1) + b1
        z1 = self.sigmoid(a1)
        a2 = l.np.dot(z1, W2) + b2
        y = self.softmax(a2)
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = l.np.argmax(y, axis=1)
        t = l.np.argmax(t, axis=1)

        accuracy = l.np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

#net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
#print(net.params['W1'].shape)
#print(net.params['b1'].shape)
#print(net.params['W2'].shape)
#print(net.params['b2'].shape)
#
#x = l.np.random.rand(100, 784)
#y = net.predict(x)
#
#x = l.np.random.rand(100, 784)
#t = l.np.random.rand(100, 10)
#grads = net.numerical_gradient(x, t)








