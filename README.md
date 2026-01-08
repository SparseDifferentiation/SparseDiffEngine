3. more tests for chain rule elementwise univariate hessian
4. in the refactor, add consts
5. multiply with one constant vector/scalar argument
6. AX where X is a matrix. Can that happen? How is that canonicalized? Maybe it can't happen.
7. Must be able to compute jacobian and hessian of A @ phi(x), so linear operator needs other code! This requires new infrastructure, I think.
8. Shortcut hessians of affine stuff?