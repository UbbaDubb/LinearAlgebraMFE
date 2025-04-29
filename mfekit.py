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