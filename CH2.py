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