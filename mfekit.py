import pandas as pd
import numpy as np

def returns(df):
    """
    Calculate the returns of a DataFrame of prices.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing price data with a 'Date' column.
    
    Returns:
    pd.DataFrame: DataFrame containing the returns.
    """
    returns = pd.DataFrame()
    returns.colnames = df.columns[1:]  # Get the column names excluding the first one (Date)

    for col in df.columns[1:]:
        # Calculate the percentage change for each column
        returns[col] = df[col].pct_change()
    
    # Drop the first row with NaN values
    returns = returns.dropna().reset_index(drop=True)

    return returns

def cov(df):
    """
    Calculate the covariance matrix of a DataFrame of returns.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing returns data.
    
    Returns:
    pd.DataFrame: Covariance matrix of the returns.
    """
    # Convert DataFrame to numpy array
    returns_mat = df.to_numpy()
    
    # Compute mean for centering
    mean_vec = np.mean(returns_mat, axis=0)  # Compute column means
    
    # Center the data by subtracting the mean
    returns_centered = returns_mat - mean_vec  # Subtract means from values in each column
    
    # Transpose
    returns_mat_t = returns_centered.T
    
    # Correct covariance formula
    cov_matrix = (1 / (len(returns_mat) - 1)) * (returns_mat_t @ returns_centered)
    
    return cov_matrix

def log_returns(df):
    """
    Calculate the log returns of a DataFrame of prices.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing price data with a 'Date' column.
    
    Returns:
    pd.DataFrame: DataFrame containing the log returns.
    """
    log_returns = np.log(df.iloc[:, 1:] / df.iloc[:, 1:].shift(1))
    
    # Drop the first row with NaN values
    log_returns = log_returns.dropna().reset_index(drop=True)

    return log_returns

def corr(df):
    """
    Calculate the correlation matrix of a DataFrame of returns.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing returns data.
    
    Returns:
    pd.DataFrame: Correlation matrix of the returns.
    """
    # Covariance matrix
    cov_matrix = cov(df)
    
    # Standard deviations
    std_devs = np.sqrt(np.diag(cov_matrix))
    
    # Inverse of diagonal matrix of standard deviations
    D_inv = np.linalg.inv(np.diag(std_devs))
    
    # Correlation matrix
    corr_matrix = D_inv @ cov_matrix @ D_inv

    return corr_matrix

def forward_susbt(L, b):
    """
    Forward substitution to solve Lx = b.
    
    Parameters:
    L (np.ndarray): Lower triangular matrix of size n x n
    b (np.ndarray): column vector of size n
    
    Returns:
    np.ndarray: Solution vector x.
    """
    n = len(b)
    x = np.zeros(n)
    
    for i in range(n):
        x[i] = (b[i] - np.dot(L[i, :i], x[:i])) / L[i, i]
    
    # Return as a column vector
    x = x.reshape(-1, 1)
    
    return x

def forward_subst_bidiag(L,b):
    """
    Forward substitution to solve Lx = b for a bidiagonal matrix L.
    
    Parameters:
    L (np.ndarray): nonsingular lower triangular Bidiagonal matrix of size n x n
    b (np.ndarray): column vector of size n
    
    Returns:
    np.ndarray: Solution vector x.
    """
    n = len(b)
    x = np.zeros(n)
    
    # Forward substitution
    for i in range(n):
        if i == 0:
            x[i] = b[i] / L[i, i]
        else:
            x[i] = (b[i] - L[i, i-1] * x[i-1]) / L[i, i]
    
    # Return as a column vector
    x = x.reshape(-1, 1)
    
    return x

def zero_rates(cash_flows, prices):
    """
    Calculate zero rates from cash flows and prices.
    
    Parameters:
    cash_flows (np.ndarray): Cash flow matrix of size n x m
    prices (np.ndarray): Price vector of size n x 1
    
    Returns:
    np.ndarray: Zero rates vector of size m x 1
    """
    discount_factors = forward_subst_bidiag(cash_flows, prices)
    zero_rates = np.zeros(cash_flows.shape[1])
    for i in range(cash_flows.shape[1]):
        zero_rates[i] = (-np.log(discount_factors[i]))/(i+1)
    
    return zero_rates

def backward_subst(U, b):
    """
    Backward substitution to solve Ux = b.
    
    Parameters:
    U (np.ndarray): nonsingular Upper triangular matrix of size n x n
    b (np.ndarray): column vector of size n
    
    Returns:
    np.ndarray: Solution vector x.
    """
    n = len(b)
    x = np.zeros(n)
    
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    
    # Return as a column vector
    x = x.reshape(-1, 1)
    
    return x

def backward_subst_bidiag(U, b):
    """
    Backward substitution to solve Ux = b for a bidiagonal matrix U.
    
    Parameters:
    U (np.ndarray): nonsingular upper triangular Bidiagonal matrix of size n x n
    b (np.ndarray): column vector of size n
    
    Returns:
    np.ndarray: Solution vector x.
    """
    n = len(b)
    x = np.zeros(n)
    
    # Backward substitution
    for i in range(n-1, -1, -1):
        if i == n-1:
            x[i] = b[i] / U[i, i]
        else:
            x[i] = (b[i] - U[i, i+1] * x[i+1]) / U[i, i]
    
    # Return as a column vector
    x = x.reshape(-1, 1)
    
    return x


def lu_no_pivoting(A):
    """
    LU decomposition without pivoting.
    
    Parameters:
    A (np.ndarray): Square matrix of size n x n
    
    Returns:
    (L, U): Tuple of lower and upper triangular matrices
    """
    A = A.copy().astype(float)  # Avoid modifying input and ensure float division
    n = A.shape[0]
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
        for j in range(i+1, n):
            if U[i, i] == 0:
                raise ZeroDivisionError(f"Zero pivot encountered at index {i}")
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]

    return L, U

def linear_solve_LU_no_pivoting(A, b):
    """
    A = nonsingular matrix of size n with LU deocmposition
    b = column vector of size n

    x = solution to Ax=b

    """
    [L,U] = lu_no_pivoting(A)

    y = forward_susbt(L, b)
    x = backward_subst(U, y)
    return x

def lu_no_pivoting_tridiag(A):
    """
    A = nonsingular tridiagonal matrix of size n with LU decomposition

    L = lower triangular matrix with entries 1 on main diagonal
    U = upper triangular matrix
    such that A = LU
    """
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    for i in range(1, (n-1)):
        if A[i-1, i] == 0:
            raise ZeroDivisionError(f"Zero pivot encountered at index {i-1}")
        L[i, i] = 1
        L[i+1, i] = A[i+1, i] / A[i, i]
        U[i,i] = A[i, i]
        U[i, i+1] = A[i,i+1]
        A[i+1, i+1] = A[i+1, i+1] - L[i+1, i] * U[i, i+1]

    L[n,n] = 1
    U[n, n] = A[n, n]

    return L, U

def linear_solve_LU_no_pivoting_tridiag(A, b):
    """
    A = nonsingular tridiagonal matrix of size n with LU decomposition
    b = column vector of size n

    x = solution to Ax=b
    """
    [L,U] = lu_no_pivoting_tridiag(A)

    y = forward_subst_bidiag(L, b)
    x = backward_subst_bidiag(U, y)
    return x

"""
def lu_row_pivoting(A):
    
    #A= nonsingular matrix of size n

    #P= permuation matrix, stored as vector of its original entries
    #L = lower triangular matrix with entries 1 on the diagonal
    #U = upper triangular matrix
    #such that PA = LU

    

    n = len(A) -1
    P = np.arange(n)  # Initialize permutation vector
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range (0, (n-1)):
    #find i_max, index of the largest entry from vector (A:n, i)
        i_max = np.argmax(np.abs(A[i:, i])) + i  # Adjust index to account for the current row

        if i_max != i:
            # Swap rows in A
            vv = A[i, :].copy()
            A[i, :] = A[i_max, :]
            A[i_max, :] = vv

            #Update permutation vector P
            cc= P[i].copy()
            P[i] = P[i_max]
            P[i_max] = cc
            
            #Switch rows i and i~_max of L
            if i> 1:
                ww = L[i, 1:(i-1)].copy()
                L[i, 1:(i-1)] = L[i_max, 1:(i-1)]
                L[i_max, 1:(i-1)] = ww
            
        for j in range (i,n):
            L[j,i] = A[j,i] / A[i,i]
            U[i,j] = A[i,j]

        for j in range(i+1, n):
            for k in range(i+1, n):
                A[j,k] = A[j,k] - L[j,i] * U[i,k]

        L[n,n] = 1
        U[n, n] = A[n, n]

    return P, L, U
"""

def lu_row_pivoting(A):
    """
    Perform LU decomposition with row pivoting.

    Parameters:
    A (np.ndarray): Nonsingular square matrix of size n x n.

    Returns:
    tuple: (P, L, U)
        P (np.ndarray): Permutation matrix as a vector of indices.
        L (np.ndarray): Lower triangular matrix with 1s on the diagonal.
        U (np.ndarray): Upper triangular matrix.
    """
    A = A.copy().astype(float)  # Ensure input matrix is not modified and is float
    n = A.shape[0]
    P = np.arange(n)  # Initialize permutation vector
    L = np.eye(n)  # Initialize L as identity matrix
    U = np.zeros_like(A)  # Initialize U as a zero matrix

    for i in range(n):
        # Find the index of the largest pivot element in the current column
        i_max = np.argmax(np.abs(A[i:, i])) + i

        if A[i_max, i] == 0:
            raise ValueError("Matrix is singular and cannot be decomposed.")

        # Swap rows in A and update the permutation vector
        if i_max != i:
            A[[i, i_max]] = A[[i_max, i]]
            P[[i, i_max]] = P[[i_max, i]]
            if i > 0:
                L[[i, i_max], :i] = L[[i_max, i], :i]

        # Update L and U matrices
        for j in range(i, n):
            U[i, j] = A[i, j]
        for j in range(i + 1, n):
            L[j, i] = A[j, i] / U[i, i]
            A[j, i:] -= L[j, i] * U[i, i:]

    return P, L, U
        
def linear_solve_lu_row_pivoting(A, b):
    """
    A = nonsingular matrix of size n wit hLU decomposition
    b = column vector of size n
    """
    P , L , U = lu_row_pivoting(A)
    y=forward_susbt(L, b[P])  # Apply permutation to b
    x = backward_subst(U, y)
    
    return x
    