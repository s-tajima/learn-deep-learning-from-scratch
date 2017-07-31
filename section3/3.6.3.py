import sys, os
from common import lib as l
from common import mnist as m
import pickle

def sigmoid(x):
    return 1 / (1 + l.np.exp(-x))

def softmax(a):
    c = l.np.max(a)
    exp_a = l.np.exp(a - c)
    sum_exp_a = l.np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def get_data():
    (x_train, t_train), (x_test, t_test) = m.load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = l.np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = l.np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = l.np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = l.np.argmax(y_batch, axis = 1)
    accuracy_cnt += l.np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
