import sys, os
from common import lib as l
from common import mnist as m
import pickle

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x

def tangent_line(f, x):
    d = numerical_diff(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y

x = l.np.arange(0.0, 20.0, 0.1)
y = function_1(x)

l.plt.xlabel("x")
l.plt.ylabel("f(x)")

tf = tangent_line(function_1, 5)
y2 = tf(x)

tf = tangent_line(function_1, 10)
y3 = tf(x)

l.plt.xlim(0, 20)
l.plt.ylim(0, 6)

l.plt.plot(x, y)
l.plt.plot(x, y2)
l.plt.plot(x, y3)

l.plt.plot([0, 5], [function_1(5), function_1(5)], ':', lw=1)
l.plt.plot([5, 5], [function_1(5), 0], ':', lw=1)

l.plt.plot([0, 10], [function_1(10), function_1(10)], ':', lw=1)
l.plt.plot([10, 10], [function_1(10), 0], ':', lw=1)
l.plt_show_alt(l.plt, "4.3.2")
