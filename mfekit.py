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
    L (np.ndarray): Bidiagonal matrix of size n x n
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