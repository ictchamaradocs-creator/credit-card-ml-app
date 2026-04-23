import pandas as pd

# Load datasets from data folder
df1 = pd.read_csv("data/Credit_Card_Dataset_2025_Sept_1.csv")
df2 = pd.read_csv("data/Credit_Card_Dataset_2025_Sept_2.csv")

# Merge datasets using ID and User columns
df = pd.merge(df1, df2, left_on="ID", right_on="User", how="inner")

# Remove duplicate rows
df = df.drop_duplicates()

# Check missing values
print(df.isnull().sum())

# Fill missing values with 0
df = df.fillna(0)

# Save cleaned dataset
df.to_csv("data/cleaned_data.csv", index=False)

print("Dataset merged and cleaned successfully!")