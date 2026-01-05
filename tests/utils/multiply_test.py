import numpy as np

A = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0], [0.0, 6.0, 0.0]])
B = np.array([[1.0, 0.0, 4], [0.0, 2.0, 7], [3.0, 0.0, 2], [0.0, 4.0, -1]])
w = np.array([1.0, 2.0, 3.0, 4.0])

H = A.T @ np.diag(w) @ B + B.T @ np.diag(w) @ A
print(H)

# print H in CSR
from scipy.sparse import csr_matrix

H_csr = csr_matrix(H)
print("H in CSR format:")
print("data:", H_csr.data)
print("indices:", H_csr.indices)
print("indptr:", H_csr.indptr)
