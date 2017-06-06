import numpy as np

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B.shape)

print(np.dot(A, B))

C = np.array([[1, 2], [3, 4]])
print(C.shape)
print(A.shape)
#print(np.dot(A, C))

D = np.array([[1, 2], [3, 4], [5, 6]])
print(D.shape)

E = np.array([7, 8])
print(E.shape)

print(np.dot(D, E))
