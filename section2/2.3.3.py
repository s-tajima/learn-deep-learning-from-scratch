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

print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))

print(NAND(0, 0))
print(NAND(1, 0))
print(NAND(0, 1))
print(NAND(1, 1))

print(OR(0, 0))
print(OR(1, 0))
print(OR(0, 1))
print(OR(1, 1))
