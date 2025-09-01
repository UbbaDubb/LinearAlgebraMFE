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

# Question 4

q4_A = np.array([
    [2, -1, 3, -1],
    [1, 0, -2, -4],
    [3, 1, 1, -2],
    [-4, 1, 0, 2]
])

q4_b = np.array([
    [-1],
    [0],
    [1],
    [2]
])

print("LU decomposition with pivoting:")
q4_x = mfk.linear_solve_lu_row_pivoting(q4_A, q4_b)
print(q4_x)

# Question 6/7

q6_A = np.array([
    [2, -1, 1],
    [-2, 1, 3],
    [4, 0, -1]
])

q6 = mfk.lu_row_pivoting(q6_A)
print("LU decomposition with row pivoting:")
print("L =\n", q6[1])
print("U =\n", q6[2])
print("P =\n", q6[0])

q7 = mfk.lu_col_pivoting(q6_A)
print("LU decomposition with column pivoting:")
print("L =\n", q7[1])
print("U =\n", q7[2])
print("P =\n", q7[0])