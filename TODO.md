4. in the refactor, add consts
10. For performance reasons, is it useful to have a dense matmul with A and B as dense matrices?
11. right matmul, add broadcasting logic as in left matmul. Is this necessary?

Going through all atoms to see that sparsity pattern is computed in initialization of hessian:
3. trace - not ok
