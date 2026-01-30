import scipy.sparse as sp
import numpy as np

np.random.seed(42)
m = 10
n = 15
# density = 0.1
# A = sp.random(m, n, density=density, format="csc", dtype=float)

Ap = np.array([0, 1, 1, 1, 1, 4, 5, 6, 7, 8, 9, 11, 11, 11, 13, 15])
Ai = np.array([5, 0, 6, 9, 0, 5, 1, 3, 6, 0, 6, 3, 6, 6, 8])
Ax = np.random.randint(1, 10, size=len(Ai))
d = np.random.randint(1, 10, size=m)
A = sp.csc_matrix((Ax, Ai, Ap), shape=(m, n))


C = A.T @ sp.diags(d) @ A

C.sort_indices()

import pdb

pdb.set_trace()
