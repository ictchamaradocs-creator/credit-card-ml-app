import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("data/cleaned_data.csv")

# Remove unnecessary column
df = df.drop("Unnamed: 0", axis=1)

# =========================
# TARGET & FEATURES
# =========================
y = df["TARGET"]

X = df.drop("TARGET", axis=1)

# =========================
# ENCODE CATEGORICAL DATA
# =========================
le = LabelEncoder()

for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = le.fit_transform(X[col].astype(str))

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# =========================
# FEATURE SCALING
# =========================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# KNN MODEL
# =====================================================
print("\n========== KNN MODEL ==========")

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train_scaled, y_train)

knn_predictions = knn_model.predict(X_test_scaled)

# Metrics
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions)
knn_recall = recall_score(y_test, knn_predictions)
knn_f1 = f1_score(y_test, knn_predictions)

print("KNN Accuracy:", knn_accuracy)
print("KNN Precision:", knn_precision)
print("KNN Recall:", knn_recall)
print("KNN F1 Score:", knn_f1)

# Classification Report
print("\nKNN Classification Report")
print(classification_report(y_test, knn_predictions))

# Confusion Matrix
knn_cm = confusion_matrix(y_test, knn_predictions)

plt.figure(figsize=(5,4))

sns.heatmap(knn_cm, annot=True, fmt='d', cmap='Blues')

plt.title("KNN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =====================================================
# NAIVE BAYES MODEL
# =====================================================
print("\n========== NAIVE BAYES MODEL ==========")

nb_model = GaussianNB()

nb_model.fit(X_train_scaled, y_train)

nb_predictions = nb_model.predict(X_test_scaled)

# Metrics
nb_accuracy = accuracy_score(y_test, nb_predictions)
nb_precision = precision_score(y_test, nb_predictions)
nb_recall = recall_score(y_test, nb_predictions)
nb_f1 = f1_score(y_test, nb_predictions)

print("Naive Bayes Accuracy:", nb_accuracy)
print("Naive Bayes Precision:", nb_precision)
print("Naive Bayes Recall:", nb_recall)
print("Naive Bayes F1 Score:", nb_f1)

# Classification Report
print("\nNaive Bayes Classification Report")
print(classification_report(y_test, nb_predictions))

# Confusion Matrix
nb_cm = confusion_matrix(y_test, nb_predictions)

plt.figure(figsize=(5,4))

sns.heatmap(nb_cm, annot=True, fmt='d', cmap='Greens')

plt.title("Naive Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =====================================================
# SELECT BEST MODEL
# =====================================================
if knn_f1 >= nb_f1:
    final_model = knn_model
    print("\nKNN selected as final model")
else:
    final_model = nb_model
    print("\nNaive Bayes selected as final model")

# =====================================================
# SAVE MODEL
# =====================================================
pickle.dump(final_model, open("models/model.pkl", "wb"))

print("\nModel trained successfully!")
print("model.pkl saved inside models folder")