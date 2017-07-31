from mpl_toolkits.mplot3d import Axes3D
import sys, os
from common import lib as l
from common import mnist as m

def function_2(x):
    return x[0]**2 + x[1]**2

x1 = l.np.arange(-3.0, 3.0, 0.1)
x2 = l.np.arange(-3.0, 3.0, 0.1)
X1, X2 = l.np.meshgrid(x1, x2)
y = function_2(l.np.array([X1, X2]))

l.plt.xlabel("x1")
l.plt.xlabel("x2")
l.plt.ylabel("f(x)")

fig = l.plt.figure()
ax = Axes3D(fig)
ax.plot_wireframe(X1,X2,y)

l.plt_show_alt(l.plt, "4.3.3")
