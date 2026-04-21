import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("cleaned_data.csv")

# ❌ remove useless column
df = df.drop("Unnamed: 0", axis=1)

# 🎯 target
y = df["TARGET"]

# features
X = df.drop("TARGET", axis=1)

# 🔄 convert text to numbers
le = LabelEncoder()

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col].astype(str))

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained successfully!")