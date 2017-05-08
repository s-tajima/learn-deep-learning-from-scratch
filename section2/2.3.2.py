from common import lib as l

x = l.np.array([0, 1])
w = l.np.array([0.5, 0.5])
b = -0.7

print(w*x)
print(l.np.sum(w*x))
print(l.np.sum(w*x) + b)
print(l.np.sum(w*x) + b > 0)


