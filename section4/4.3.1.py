import sys, os
from common import lib as l
from common import mnist as m
import pickle

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

f = lambda x: x * 2

result = numerical_diff(f, 1)
print(result)
