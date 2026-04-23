from shiny import App, ui, render
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# ================= LOAD MODEL =================
model = pickle.load(open("models/model.pkl", "rb"))

# ================= LOAD DATASET =================
df = pd.read_csv("data/cleaned_data.csv", low_memory=False)

# Remove unnecessary column
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# ================= UI =================
app_ui = ui.page_fluid(

    ui.h2("💳 Credit Card Approval Prediction"),
    ui.hr(),
    ui.h3("Applicant Information"),
    ui.input_numeric("age", "Age", 30),
    ui.input_numeric("income", "Income", 50000),
    ui.input_numeric("family_size", "Family Size", 2),

    ui.input_select("gender", "Gender", ["M", "F"]),
    ui.input_select("car", "Own Car", ["Y", "N"]),
    ui.input_select("house", "Own House", ["Y", "N"]),
    ui.input_action_button("predict_btn", "Predict Approval"),
    ui.hr(),

    ui.br(),

    ui.h4("Prediction Result"),
    ui.output_text("result"),
    ui.output_text("confidence"),

    ui.hr(),

    ui.h4("📊 Dataset Insights"),

    ui.output_plot("approval_chart"),
    ui.output_plot("age_chart"),
    ui.output_plot("income_chart"),
    ui.output_plot("age_boxplot"),

    ui.hr(),

    ui.h4("📊 Model Performance"),

    ui.output_text("accuracy"),
    ui.output_text("confusion")
    
)

# ================= SERVER =================
def server(input, output, session):

    # ================= PREPARE INPUT DATA =================
    def prepare_data():

        sample = df.drop("TARGET", axis=1).iloc[0:1].copy()

        # User inputs
        sample["AGE"] = input.age()
        sample["INCOME"] = input.income()
        sample["FAMILY SIZE"] = input.family_size()

        sample["GENDER"] = input.gender()
        sample["CAR"] = input.car()
        sample["REALITY"] = input.house()

        # Encode text columns
        for col in sample.columns:
            if sample[col].dtype == "object":
                sample[col] = sample[col].astype("category").cat.codes

        return sample

    # ================= PREDICTION RESULT =================
    @output
    @render.text
    def result():
        try:
            sample = prepare_data()

            prediction = model.predict(sample)[0]

            if prediction == 1:
                return "✅ Approved"
            else:
                return "❌ Not Approved"

        except Exception as e:
            return f"Error: {str(e)}"

    # ================= CONFIDENCE SCORE =================
    @output
    @render.text
    def confidence():

        try:
            sample = prepare_data()

            prob = model.predict_proba(sample)[0][1]

            return f"Confidence Score: {prob:.2f}"

        except:
            return ""

    # ================= APPROVAL DISTRIBUTION =================
    @output
    @render.plot
    def approval_chart():

        counts = df["TARGET"].value_counts()

        labels = ["Not Approved", "Approved"]

        values = [
            counts.get(0, 0),
            counts.get(1, 0)
        ]

        plt.figure(figsize=(5,4))

        plt.bar(labels, values)

        plt.title("Approval Distribution")

        plt.xlabel("Status")
        plt.ylabel("Count")

        return plt.gcf()

    # ================= AGE DISTRIBUTION =================
    @output
    @render.plot
    def age_chart():

        plt.figure(figsize=(5,4))

        plt.hist(df["AGE"], bins=20)

        plt.title("Age Distribution")

        plt.xlabel("Age")
        plt.ylabel("Frequency")

        return plt.gcf()

    # ================= INCOME DISTRIBUTION =================
    @output
    @render.plot
    def income_chart():

        plt.figure(figsize=(5,4))

        plt.hist(df["INCOME"], bins=20)

        plt.title("Income Distribution")

        plt.xlabel("Income")
        plt.ylabel("Frequency")

        return plt.gcf()

    # ================= AGE BOXPLOT =================
    @output
    @render.plot
    def age_boxplot():

        plt.figure(figsize=(5,4))

        plt.boxplot(df["AGE"])

        plt.title("Age Boxplot")

        plt.ylabel("Age")

        return plt.gcf()

    # ================= MODEL ACCURACY =================
    @output
    @render.text
    def accuracy():

        try:

            X = df.drop("TARGET", axis=1)
            y = df["TARGET"]

            # Encode text columns
            for col in X.columns:
                if X[col].dtype == "object":
                    X[col] = X[col].astype("category").cat.codes

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)

            return f"Model Accuracy: {acc:.2f}"

        except Exception as e:
            return f"Error: {str(e)}"

    # ================= CONFUSION MATRIX =================
    @output
    @render.text
    def confusion():

        try:

            X = df.drop("TARGET", axis=1)
            y = df["TARGET"]

            # Encode text columns
            for col in X.columns:
                if X[col].dtype == "object":
                    X[col] = X[col].astype("category").cat.codes

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            preds = model.predict(X_test)

            cm = confusion_matrix(y_test, preds)

            return f"Confusion Matrix:\n{cm}"

        except Exception as e:
            return f"Error: {str(e)}"

# ================= APP =================
app = App(app_ui, server)