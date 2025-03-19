import pandas as pd

# Define file path
file_path = "data/indeces-jul26-aug9-2012.xlsx"

# Read the Excel file
df = pd.read_excel(file_path, engine="openpyxl")  # Specify openpyxl for .xlsx files

# Display the first few rows
print(df)

returns = pd.DataFrame()

returns['DJ_%'] = df['Dow Jones'].pct_change()
returns['NASDAQ_%'] = df['NASDAQ '].pct_change()
returns['S&P500_%'] = df['S&P 500'].pct_change()

returns = returns.iloc[1:].reset_index(drop=True)

print(returns)