from shiny import App, ui, render
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Load model
model = pickle.load(open("model.pkl", "rb"))

# ================= UI =================
app_ui = ui.page_fluid(
    ui.h2("💳 Credit Card Approval Prediction"),

    ui.input_numeric("age", "Age", 30),
    ui.input_numeric("income", "Income", 50000),
    ui.input_numeric("family_size", "Family Size", 2),

    ui.input_select("gender", "Gender", ["M", "F"]),
    ui.input_select("car", "Own Car", ["Y", "N"]),
    ui.input_select("house", "Own House", ["Y", "N"]),

    ui.br(),
    ui.h4("Result:"),
    ui.output_text("result"),
    ui.output_text("confidence"),

    ui.hr(),
    ui.h4("📊 Dataset Insights"),

    ui.output_plot("approval_chart"),
    ui.output_plot("age_chart"),
    ui.output_plot("income_chart"),
    ui.output_plot("age_boxplot"),   # ✅ added

    ui.hr(),
    ui.h4("📊 Model Performance"),
    ui.output_text("accuracy"),
    ui.output_text("confusion")
)

# ================= SERVER =================
def server(input, output, session):

    def prepare_data():
        df = pd.read_csv("cleaned_data.csv", low_memory=False)
        df = df.drop(columns=["Unnamed: 0"], errors="ignore")

        sample = df.drop("TARGET", axis=1).iloc[0:1].copy()

        sample["AGE"] = input.age()
        sample["INCOME"] = input.income()
        sample["FAMILY SIZE"] = input.family_size()
        sample["GENDER"] = input.gender()
        sample["CAR"] = input.car()
        sample["REALITY"] = input.house()

        for col in sample.columns:
            if sample[col].dtype == "object":
                sample[col] = sample[col].astype("category").cat.codes

        return sample

    @output
    @render.text
    def result():
        try:
            sample = prepare_data()
            prediction = model.predict(sample)[0]
            return "✅ Approved" if prediction == 1 else "❌ Not Approved"
        except Exception as e:
            return f"Error: {str(e)}"

    @output
    @render.text
    def confidence():
        try:
            sample = prepare_data()
            prob = model.predict_proba(sample)[0][1]
            return f"Confidence: {prob:.2f}"
        except:
            return ""

    # 📊 Approval chart
    @output
    @render.plot
    def approval_chart():
        df = pd.read_csv("cleaned_data.csv", low_memory=False)
        counts = df["TARGET"].value_counts()

        labels = ["Not Approved", "Approved"]
        values = [counts.get(0, 0), counts.get(1, 0)]

        plt.figure()
        plt.bar(labels, values)
        plt.title("Approval Distribution")
        plt.xlabel("Status")
        plt.ylabel("Count")
        return plt.gcf()

    # 📊 Age distribution
    @output
    @render.plot
    def age_chart():
        df = pd.read_csv("cleaned_data.csv", low_memory=False)

        plt.figure()
        plt.hist(df["AGE"], bins=20)
        plt.title("Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        return plt.gcf()

    # 📊 Income distribution
    @output
    @render.plot
    def income_chart():
        df = pd.read_csv("cleaned_data.csv", low_memory=False)

        plt.figure()
        plt.hist(df["INCOME"], bins=20)
        plt.title("Income Distribution")
        plt.xlabel("Income")
        plt.ylabel("Frequency")
        return plt.gcf()

    # 📊 Age Boxplot (NEW)
    @output
    @render.plot
    def age_boxplot():
        df = pd.read_csv("cleaned_data.csv", low_memory=False)

        plt.figure()
        plt.boxplot(df["AGE"])
        plt.title("Age Boxplot (Outliers)")
        plt.ylabel("Age")
        return plt.gcf()

    # 📊 Accuracy
    @output
    @render.text
    def accuracy():
        try:
            df = pd.read_csv("cleaned_data.csv", low_memory=False)
            df = df.drop(columns=["Unnamed: 0"], errors="ignore")

            X = df.drop("TARGET", axis=1)
            y = df["TARGET"]

            for col in X.columns:
                if X[col].dtype == "object":
                    X[col] = X[col].astype("category").cat.codes

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)

            return f"Accuracy: {acc:.2f}"
        except Exception as e:
            return f"Error: {str(e)}"

    # 📊 Confusion Matrix
    @output
    @render.text
    def confusion():
        try:
            df = pd.read_csv("cleaned_data.csv", low_memory=False)
            df = df.drop(columns=["Unnamed: 0"], errors="ignore")

            X = df.drop("TARGET", axis=1)
            y = df["TARGET"]

            for col in X.columns:
                if X[col].dtype == "object":
                    X[col] = X[col].astype("category").cat.codes

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            preds = model.predict(X_test)
            cm = confusion_matrix(y_test, preds)

            return f"Confusion Matrix:\n{cm}"
        except:
            return ""

# ================= APP =================
app = App(app_ui, server)