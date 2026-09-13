import pandas as pd

# Load the dataset
df = pd.read_csv("Dataset/cardio_train.csv", sep=";")

print("========== ORIGINAL DATASET ==========")
print("Rows and columns:", df.shape)

# Check unique values in categorical/binary columns
print("\n========== UNIQUE VALUES ==========")

columns_to_check = [
    "gender",
    "cholesterol",
    "gluc",
    "smoke",
    "alco",
    "active",
    "cardio"
]

for column in columns_to_check:
    print(f"{column}: {sorted(df[column].unique())}")

# Check minimum and maximum values
print("\n========== MINIMUM VALUES ==========")
print(df.min(numeric_only=True))

print("\n========== MAXIMUM VALUES ==========")
print(df.max(numeric_only=True))

# Check duplicate rows
print("\n========== DUPLICATES ==========")
print("Duplicate rows:", df.duplicated().sum())
# Check suspicious blood pressure values
print("\n========== SUSPICIOUS BLOOD PRESSURE VALUES ==========")

print("ap_hi below 60:", (df["ap_hi"] < 60).sum())
print("ap_hi above 250:", (df["ap_hi"] > 250).sum())

print("ap_lo below 40:", (df["ap_lo"] < 40).sum())
print("ap_lo above 200:", (df["ap_lo"] > 200).sum())

# Check whether systolic pressure is lower than diastolic pressure
print(
    "ap_hi <= ap_lo:",
    (df["ap_hi"] <= df["ap_lo"]).sum()
)
# ==========================================
# CLEANING INVALID BLOOD PRESSURE VALUES
# ==========================================

# Keep only reasonable blood pressure values
df_clean = df[
    (df["ap_hi"] >= 60) &
    (df["ap_hi"] <= 250) &
    (df["ap_lo"] >= 40) &
    (df["ap_lo"] <= 200) &
    (df["ap_hi"] > df["ap_lo"])
].copy()

print("\n========== AFTER BLOOD PRESSURE CLEANING ==========")
print("Original rows:", len(df))
print("Rows after cleaning:", len(df_clean))
print("Rows removed:", len(df) - len(df_clean))
# ==========================================
# CONVERT AGE FROM DAYS TO YEARS
# ==========================================

df_clean["age_years"] = (df_clean["age"] / 365.25).round().astype(int)

print("\n========== AGE CONVERSION ==========")
print(df_clean[["age", "age_years"]].head())

print("\nAge range in years:")
print(df_clean["age_years"].min(), "to", df_clean["age_years"].max())
# ==========================================
# FINALIZE CLEANED DATASET
# ==========================================

# Remove ID because it is only an identifier
df_clean = df_clean.drop(columns=["id", "age"])

# Check missing values again
print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df_clean.isnull().sum())

# Display final columns
print("\n========== FINAL COLUMNS ==========")
print(df_clean.columns.tolist())

# Display final dataset shape
print("\n========== FINAL DATASET SHAPE ==========")
print(df_clean.shape)

# Save cleaned dataset
df_clean.to_csv(
    "Output/cardio_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")