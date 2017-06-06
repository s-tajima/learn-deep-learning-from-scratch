from common import lib as l

def step_function_1(x):
    if x > 0:
        return 1
    else:
        return 0

def step_function_2(x):
    y = x > 0
    return y.astype(l.np.int)

print(step_function_1(1))
print(step_function_1(-1))
print(step_function_1(0))
print(step_function_1(10))

print(step_function_2(np.array([1.0, 2.0])))
