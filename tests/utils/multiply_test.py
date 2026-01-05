import numpy as np
import scipy.sparse as sp

# test 1
A = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0], [0.0, 6.0, 0.0]])
B = np.array([[1.0, 0.0, 4], [0.0, 2.0, 7], [3.0, 0.0, 2], [0.0, 4.0, -1]])
w = np.array([1.0, 2.0, 3.0, 4.0])
H = A.T @ np.diag(w) @ B + B.T @ np.diag(w) @ A
print(H)
H_csr = sp.csr_matrix(H)
print("H in CSR format:")
print("data:", H_csr.data)
print("indices:", H_csr.indices)
print("indptr:", H_csr.indptr)

# test 2
m = 5
n = 10
density = 0.2
A = sp.random(m, n, density=density, format="csr", data_rvs=np.random.randn)
B = sp.random(m, n, density=density, format="csr", data_rvs=np.random.randn)
w = np.random.rand(m)
H = A.T @ sp.diags(w) @ B + B.T @ sp.diags(w) @ A
H = H.tocsr()
print("Random sparse H in CSR format:")
print("data:", H.data)
print("indices:", H.indices)
print("indptr:", H.indptr)

print("A in csr:")
print("data:", A.data)
print("indices:", A.indices)
print("indptr:", A.indptr)

print("B in csr:")
print("data:", B.data)
print("indices:", B.indices)
print("indptr:", B.indptr)

print("w:", w)
