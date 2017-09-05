from mpl_toolkits.mplot3d import Axes3D
import sys, os
from common import lib as l
from common import mnist as m
import pickle


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


def function_2(x):
    if x.ndim == 1:
        return l.np.sum(x**2)
    else:
        return l.np.sum(x**2, axis=1)


def tangent_line(f, x):
    d = numerical_gradient(f, x)
    print(d)
    y = f(x) - d*x
    return lambda t: d*t + y
     
if __name__ == '__main__':
    x0 = l.np.arange(-2, 2.5, 0.25)
    x1 = l.np.arange(-2, 2.5, 0.25)
    X, Y = l.np.meshgrid(x0, x1)
    
    X = X.flatten()
    Y = Y.flatten()
    
    grad = numerical_gradient(function_2, l.np.array([X, Y]) )
    
    l.plt.figure()
    l.plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666", label="...")#,headwidth=10,scale=40,color="#444444")
    l.plt.xlim([-2, 2])
    l.plt.ylim([-2, 2])
    l.plt.xlabel('x0')
    l.plt.ylabel('x1')
    l.plt.grid()
    l.plt.legend()
    l.plt.draw()
    l.plt_show_alt(l.plt, "4.4")
