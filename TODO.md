3. more tests for chain rule elementwise univariate hessian
4. in the refactor, add consts
6. AX where X is a matrix. Can that happen? How is that canonicalized? Maybe it can't happen.
7. Must be able to compute jacobian and hessian of A @ phi(x), so linear operator needs other code! This requires new infrastructure, I think.
8. Shortcut hessians of affine stuff?
10. For performance reasons, is it useful to have a dense matmul with A and B as dense matrices?
11. right matmul, add broadcasting logic as in left matmul

Going through all atoms to see that sparsity pattern is computed in initialization of jacobian:
2. trace - not ok

Going through all atoms to see that sparsity pattern is computed in initialization of hessian:
2. hstack - not ok
3. trace - not ok
