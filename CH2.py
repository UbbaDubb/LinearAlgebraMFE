import numpy as np
import pandas as pd
import mfekit as mfk

from scipy.linalg import lu

# Define the matrix
A = np.array([
    [2, -1, 0, 1],
    [-2, 0, 1, -1],
    [4, -1, 0, 1],
    [4, -3, 0, 2]
])

# Perform LU decomposition with partial pivoting: PA = LU
P, L, U = lu(A)

print(P, "\n",  L, "\n", U)

LL, UU = mfk.lu_no_pivoting(A)
print("\nLU decomposition without pivoting:")
print(LL, "\n", UU)


B = np.array([
    [2, 0, 0],
    [4, 5, 0],
    [1,3, 7]
    ])

b = np.array([
    [6],
    [19],
    [31]
])

ans = mfk.forward_susbt(B,b)

cash_flows = np.array([
    [100, 0, 0, 0],
    [6, 106, 0, 0],
    [8, 8, 108, 0],
    [5, 5, 5, 105]
])

prices  = np.array([
    [98],
    [104],
    [111],
    [102]
])

discount_factors = mfk.forward_susbt(cash_flows, prices)
print(discount_factors)

rates = mfk.zero_rates(cash_flows, prices)
print(rates)

B = np.array([
    [2, 3, -1],
    [0, 1, 4],
    [0,0, 5]
    ])

b= np.array([
    [5],
    [6],
    [10]
])
y = mfk.backward_subst(B, b)
print(y)

C = np.array([
    [2, -1, 3, 0],
    [-4, 5, -7, -2],
    [-2, 10, -4, -7],
    [4, -14, 8, 10]
    ])

LL, UU = mfk.lu_no_pivoting(C)

print("LU decomposition without pivoting:")
print(LL, "\n", UU, "\n")
