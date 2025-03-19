import pandas as pd

# Define file path
file_path = "data/indeces-jul26-aug9-2012.xlsx"

# Read the Excel file
df = pd.read_excel(file_path, engine="openpyxl")  # Specify openpyxl for .xlsx files

# Display the first few rows
print(df.head())