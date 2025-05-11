import pandas as pd
import numpy as np

# Read the dataset
print("Loading dataset...")
df = pd.read_csv('data/Dataset.csv')

# Display basic information
print("\nDataset Shape:", df.shape)
print("\nColumn Names:")
for col in df.columns:
    print(f"- {col}")

print("\nFirst 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics:")
print(df.describe()) 