import pandas as pd
import numpy as np
import mfekit as mfk

# Define file path
file_path = "data/indeces-jul26-aug9-2012.xlsx"

# Read the Excel file
prices = pd.read_excel(file_path, engine="openpyxl")  # Specify openpyxl for .xlsx files

# Display the first few rows
print(prices)

returns = pd.DataFrame()

returns['DJ_%'] = prices['Dow Jones'].pct_change()
returns['NASDAQ_%'] = prices['NASDAQ '].pct_change()
returns['S&P500_%'] = prices['S&P 500'].pct_change()

returns = returns.iloc[1:].reset_index(drop=True)

print(returns)

print(mfk.returns(prices))

returns_mat = returns.to_numpy() #TX
# Compute mean for centering
mean_vec = np.mean(returns_mat, axis=0)  # Compute column means
returns_centered = returns_mat - mean_vec  # Subtract means from values in each column

# Transpose
returns_mat_t = returns_centered.T

# Correct covariance formula
cov = (1 / (len(returns_mat) - 1)) * (returns_mat_t @ returns_centered)

print("\nCovariance Matrix (Manual Computation):\n", cov)

cov_matrix = returns.cov()
print("\nCovariance Matrix:\n", cov_matrix)

log_returns = np.log(prices.iloc[:, 1:] / prices.iloc[:, 1:].shift(1))
log_returns = log_returns.iloc[1:].reset_index(drop=True)
print("\n",log_returns)

log_returns_mat = log_returns.to_numpy() #LX
# Compute mean for centering
log_mean_vec = np.mean(log_returns_mat, axis=0) #Compute column means
logrets_centred = log_returns_mat - log_mean_vec # Subtract means from values in each column

#Transpose
log_returns_mat_t = log_returns_mat.T

#Covariance
log_cov = (1/(len(log_returns_mat)-1)) * (log_returns_mat_t @ log_returns_mat)

print("\nCovariance MAtrix for Log Returns (Manual):\n", log_cov)

log_cov_mat = log_returns.cov()
print("Covariaqnce Matrix for Log Returns (Pandas):\n", log_cov_mat)

#std_devs = np.sqrt(np.diag(cov))
#D_inv = np.linalg.inv(np.diag(std_devs))
#corr_matrix = D_inv @ cov @ D_inv
#print("Correlation Matrix:\n", corr_matrix)

#corr_matrix = returns.corr()
#print("Correlation Matrix:\n", corr_matrix)

#corr_matrix_np = np.corrcoef(returns_mat, rowvar=False)
#print("Correlation Matrix (NumPy):\n", corr_matrix_np)