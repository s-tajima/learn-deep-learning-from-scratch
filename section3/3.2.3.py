from common import lib as l

def step_function(x):
    return l.np.array(x > 0, dtype=l.np.int)

x = l.np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

l.plt.plot(x, y)
l.plt.ylim(-0.1, 1.1)
l.plt_show_alt(l.plt)
