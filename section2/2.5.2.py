from common import lib as l

def AND(x1, x2):
    x = l.np.array([x1, x2])
    w = l.np.array([0.5, 0.5])
    b = -0.7
    tmp = l.np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = l.np.array([x1, x2])
    w = l.np.array([-0.5, -0.5])
    b = 0.7
    tmp = l.np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = l.np.array([x1, x2])
    w = l.np.array([0.5, 0.5])
    b = -0.2
    tmp = l.np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

print(XOR(0, 0))
print(XOR(0, 1))
print(XOR(1, 0))
print(XOR(1, 1))
