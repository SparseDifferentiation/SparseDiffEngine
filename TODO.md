4. in the refactor, add consts
10. For performance reasons, is it useful to have a dense matmul with A and B as dense matrices?
11. right matmul, add broadcasting logic as in left matmul

Going through all atoms to see that sparsity pattern is computed in initialization of jacobian:
2. trace - not ok

Going through all atoms to see that sparsity pattern is computed in initialization of hessian:
2. hstack - not ok
3. trace - not ok
