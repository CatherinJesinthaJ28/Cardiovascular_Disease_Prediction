import pandas as pd

# Load the dataset
df = pd.read_csv("Dataset/cardio_train.csv", sep=";")

# 1. Display first 5 records
print("========== FIRST 5 RECORDS ==========")
print(df.head())

# 2. Display dataset shape
print("\n========== DATASET SHAPE ==========")
print(df.shape)

# 3. Display column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# 4. Display data types
print("\n========== DATA TYPES ==========")
print(df.dtypes)

# 5. Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 6. Display statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())
# 7. Check for duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print("Number of duplicate rows:", df.duplicated().sum())