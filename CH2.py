import numpy as np
import pandas as pd


import numpy as np

def solve_tridiagonal(n):
    # Step size
    h = 1 / (n + 1)
    
    # Initialize tridiagonal matrix A
    main_diag = (h**2 - 2) * np.ones(n)  # Main diagonal
    upper_diag = (1 + h) * np.ones(n - 1)  # Upper diagonal
    lower_diag = (1 - h) * np.ones(n - 1)  # Lower diagonal
    
    # Construct the right-hand side vector b
    b = np.zeros(n)
    b[0] = -2 * (1 - h)  # First equation includes boundary condition y(0) = 2
    b[-1] = - (1 + h) * (3 / np.e)  # Last equation includes y(1) = 3/e
    
    # Solve using Thomas Algorithm (Specialized for tridiagonal systems)
    # Forward elimination
    for i in range(1, n):
        factor = lower_diag[i - 1] / main_diag[i - 1]
        main_diag[i] -= factor * upper_diag[i - 1]
        b[i] -= factor * b[i - 1]
    
    # Back substitution
    y = np.zeros(n)
    y[-1] = b[-1] / main_diag[-1]
    for i in range(n - 2, -1, -1):
        y[i] = (b[i] - upper_diag[i] * y[i + 1]) / main_diag[i]
    
    # Return solution
    return y

# Example: Solve for n = 4
n = 4
y_solution = solve_tridiagonal(n)
print("Solution y at interior points:", y_solution)
