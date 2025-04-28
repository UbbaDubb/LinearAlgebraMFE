import numpy as np

def solve_tridiagonal(n):
    # Define domain and step size
    a, b = 0, 1  # Adjusted domain
    h = (a - b) / n  # Correct step size
    num_points = n - 1  # Number of interior points
    x_values = np.array([a + (i+1) * h for i in range(num_points)])  # Interior points
    
    # Define diagonals
    main_diag = (h**2 + 2) * np.ones(num_points)  # Main diagonal
    upper_diag = -1 * np.ones(num_points - 1)  # Upper diagonal
    lower_diag = -1 * np.ones(num_points - 1)  # Lower diagonal
    
    # Construct right-hand side vector b
    b = np.zeros(num_points)
    b[0] = 1  # First equation with y(-1)
    b[-1] = np.e  # Last equation with y(1)
    
    # Solve using Thomas Algorithm
    for i in range(1, num_points):
        factor = lower_diag[i - 1] / main_diag[i - 1]
        main_diag[i] -= factor * upper_diag[i - 1]
        b[i] -= factor * b[i - 1]
    
    # Back substitution
    y = np.zeros(num_points)
    y[-1] = b[-1] / main_diag[-1]
    for i in range(num_points - 2, -1, -1):
        y[i] = (b[i] - upper_diag[i] * y[i + 1]) / main_diag[i]
    
    return x_values, y

# Solve for n = 8
n = 8
x_solution, y_solution = solve_tridiagonal(n)

# Print results in table format
print(f"{'i':<5} {'x_i':<10} {'y_i':<15}")
print("-" * 30)
for i, (x, y) in enumerate(zip(x_solution, y_solution), start=1):
    print(f"{i:<5} {x:<10.3f} {y:<15.10f}")
