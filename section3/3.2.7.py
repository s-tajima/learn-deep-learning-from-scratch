from common import lib as l

def relu(x):
    return l.np.maximum(0, x)

x = l.np.arange(-6.0, 6.0, 0.1)
y = relu(x)

l.plt.plot(x, y)
l.plt.ylim(-0.1, 6.0)
l.plt_show_alt(l.plt)
