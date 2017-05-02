import numpy as np

# 1.5.2
x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))

# 1.5.3
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x + y)
print(x * y)
print(x / y)

print(x / 2.0)
print(2.0 / x)

# 1.5.4
A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)
print(A.dtype)

B = np.array([[3, 0], [0, 6]])
print(A + B)
print(A * B)

print(A * 10)

# 1.5.5
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print(A * B)

A = np.array([np.array([1, 2]), np.array([3, 4])])
B = np.array([np.array([10, 20]), np.array([30])])
print(A * B)

A = np.array([[1, 2], [3, 4]])
B = np.array([[10], [20]])
print(A * B)


# 1.5.6
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0])
print(X[0][1])

for row in X:
    print(row)

X = X.flatten()
print(X)
print(X[np.array([0, 2, 4])])
print(X > 15)
print(X[X > 15])

C = np.delete(X, 0) > 15
print(X[np.append(C, True)])


