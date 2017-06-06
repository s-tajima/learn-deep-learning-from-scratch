from common import lib as l

def sigmoid(x):
    return 1 / (1 + l.np.exp(-x))

x = l.np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

l.plt.plot(x, y)
l.plt.ylim(-0.1, 1.1)
l.plt_show_alt(l.plt)
