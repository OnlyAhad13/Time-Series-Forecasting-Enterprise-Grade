import pandas as pd
import numpy as np
from datetime import datetime

# Load all datasets
train = pd.read_csv('data/raw/train.csv')
features = pd.read_csv('data/raw/features.csv')
stores = pd.read_csv('data/raw/stores.csv')

print("Train shape:", train.shape)
print("Features shape:", features.shape)
print("Stores shape:", stores.shape)

# Check the data
print("\nTrain columns:", train.columns.tolist())
# ['Store', 'Dept', 'Date', 'Weekly_Sales', 'IsHoliday']

print("\nFeatures columns:", features.columns.tolist())
# ['Store', 'Date', 'Temperature', 'Fuel_Price', 'MarkDown1-5', 'CPI', 'Unemployment', 'IsHoliday']

print("\nStores columns:", stores.columns.tolist())
# ['Store', 'Type', 'Size']

# Merge all data
df = train.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
df = df.merge(stores, on='Store', how='left')

print("\nMerged shape:", df.shape)

# Convert date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Create unique series identifier (Store-Department combination)
df['series_id'] = df['Store'].astype(str) + '_' + df['Dept'].astype(str)

# Sort by series and date
df = df.sort_values(['series_id', 'Date']).reset_index(drop=True)

# Handle missing values in features
# MarkDown columns have many NaNs (only exist during promotional periods)
markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
df[markdown_cols] = df[markdown_cols].fillna(0)  # No markdown = 0

# Fill other missing values
df['CPI'] = df.groupby('Store')['CPI'].fillna(method='ffill')
df['Unemployment'] = df.groupby('Store')['Unemployment'].fillna(method='ffill')
df['Temperature'] = df.groupby('Store')['Temperature'].fillna(method='ffill')
df['Fuel_Price'] = df.groupby('Store')['Fuel_Price'].fillna(method='ffill')

# Remove any remaining NaNs
df = df.dropna()

# Rename columns for our framework
df = df.rename(columns={
    'Date': 'timestamp',
    'Weekly_Sales': 'sales',
    'Type': 'store_type',
    'Size': 'store_size'
})

# Check for negative sales (data quality issue in Walmart dataset)
print(f"\nNegative sales entries: {(df['sales'] < 0).sum()}")
# Handle negative sales (rare, but exist in this dataset)
df = df[df['sales'] >= 0]

# Save processed data
df.to_csv('data/processed/walmart_panel.csv', index=False)
print(f"\n✓ Saved to data/processed/walmart_panel.csv")