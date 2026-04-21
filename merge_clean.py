import pandas as pd

df1 = pd.read_csv("Credit_Card_Dataset_2025_Sept_1.csv")
df2 = pd.read_csv("Credit_Card_Dataset_2025_Sept_2.csv")

df = pd.concat([df1, df2], ignore_index=True)

df = df.drop_duplicates()

print(df.isnull().sum())

df = df.fillna(0)

df.to_csv("cleaned_data.csv", index=False)

print("Done!")