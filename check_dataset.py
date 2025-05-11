import pandas as pd

# Read the dataset
df = pd.read_csv('data/Dataset.csv')

# Display first 5 rows and column information
print("\nFirst 5 rows of the dataset:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nColumn names:")
print(df.columns.tolist()) 