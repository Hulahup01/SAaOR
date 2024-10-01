import numpy as np
from copy import deepcopy


def solve(param_P, param_Q, param_A):
    m, n = param_A.shape
    P = deepcopy(param_P)
    Q = deepcopy(param_Q)
    A = deepcopy(param_A)

    print(f"P:\n{P}\n")
    print(f"Q:\n{Q}\n")
    print(f"A:\n{A}\n")

    B = np.zeros((P, Q + 1), dtype=int)
    C = np.zeros((P, Q + 1), dtype=int)

    for p in range(P):
        for q in range(Q + 1):
            if p == 0:
                B[p, q] = A[p, q]
                C[p, q] = q
            else:
                B[p, q], C[p, q] = max((A[p, i] + B[p - 1, q - i], i) for i in range(q + 1))
  
    print(f"B:\n{B}\n")
    print(f"C:\n{C}\n")

            

if __name__ == "__main__":
    P = 3
    Q = 3
    A = np.array([
        [0, 1, 2, 3],
        [0, 0, 1, 2],
        [0, 2, 2, 3]
    ])
    solve(P, Q, A)