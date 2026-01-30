import numpy as np
import scipy.sparse as sp

m = 20
n = 30
density = 0.2
A = sp.random(m, n, density=density, format="csr", dtype=float)
A.data = np.random.randint(1, 10, size=A.nnz)

Ap = A.indptr
Ai = A.indices
Ax = A.data

print("CSR matrix A:")
print("Ap:", Ap)
print("Ai:", Ai)
print("Ax:", Ax)
print("Annz:", A.nnz)

A_csc = A.tocsc()

Cp = A_csc.indptr
Ci = A_csc.indices
Cx = A_csc.data

print("CSC matrix A_csc:")
print("Cp:", Cp)
print("Ci:", Ci)
print("Cx:", Cx)
print("len(Ci):", len(Ci))
print("len(Cp):", len(Cp))
