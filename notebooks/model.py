import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("cleaned_data.csv")

# Remove useless column
df = df.drop("Unnamed: 0", axis=1)

# Target column
y = df["TARGET"]

# Features
X = df.drop("TARGET", axis=1)

# Convert text columns to numbers
le = LabelEncoder()

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col].astype(str))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Scale data for KNN
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# KNN MODEL
# =========================
knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train_scaled, y_train)

knn_predictions = knn_model.predict(X_test_scaled)

knn_accuracy = accuracy_score(y_test, knn_predictions)

print("KNN Accuracy:", knn_accuracy)

# =========================
# NAIVE BAYES MODEL
# =========================
nb_model = GaussianNB()

nb_model.fit(X_train_scaled, y_train)

nb_predictions = nb_model.predict(X_test_scaled)

nb_accuracy = accuracy_score(y_test, nb_predictions)

print("Naive Bayes Accuracy:", nb_accuracy)

# =========================
# SELECT BEST MODEL
# =========================
if knn_accuracy >= nb_accuracy:
    final_model = knn_model
    print("KNN selected as final model")
else:
    final_model = nb_model
    print("Naive Bayes selected as final model")

# Save final model
pickle.dump(final_model, open("model.pkl", "wb"))

print("Model trained successfully!")
print("Check if model.pkl created")