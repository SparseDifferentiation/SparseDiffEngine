import numpy as np
import scipy.sparse as sp


A = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
x = np.array([1, 2, 3, 4, 5])
w = np.array([-1, -2, -3])
D = np.diag(-w / ((A @ x) ** 2))
expected_hess = A.T @ D @ A
hess_csr = sp.csr_matrix(expected_hess)
print("Expected Hessian in CSR format:")
print("Data:", hess_csr.data)
print("Indices:", hess_csr.indices)
print("Indptr:", hess_csr.indptr)

A_csr = sp.csr_matrix(A)
print("A in CSR format:")
print("Data:", A_csr.data)
print("Indices:", A_csr.indices)
print("Indptr:", A_csr.indptr)
