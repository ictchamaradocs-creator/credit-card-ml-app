from shiny import App, ui, render
import pickle
import pandas as pd

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# ================= UI =================
app_ui = ui.page_fluid(
    ui.h2("Credit Card Approval Prediction System"),

    # Numeric inputs
    ui.input_numeric("age", "Age", 30),
    ui.input_numeric("income", "Income", 50000),
    ui.input_numeric("family_size", "Family Size", 2),

    # Dropdown inputs (IMPORTANT for dataset features)
    ui.input_select("gender", "Gender", ["M", "F"]),
    ui.input_select("car", "Own Car", ["Y", "N"]),
    ui.input_select("reality", "Own House", ["Y", "N"]),

    ui.hr(),

    ui.output_text("result")
)

# ================= SERVER =================
def server(input, output, session):

    @output
    @render.text
    def result():
        try:
            # Load dataset structure
            df = pd.read_csv("cleaned_data.csv")
            df = df.drop(columns=["Unnamed: 0"], errors="ignore")

            # Create sample with correct structure
            sample = df.drop("TARGET", axis=1).iloc[0:1].copy()

            # User inputs
            sample["AGE"] = input.age()
            sample["INCOME"] = input.income()
            sample["FAMILY SIZE"] = input.family_size()
            sample["GENDER"] = input.gender()
            sample["CAR"] = input.car()
            sample["REALITY"] = input.reality()

            # Convert categorical → numeric
            for col in sample.columns:
                if sample[col].dtype == "object":
                    sample[col] = sample[col].astype("category").cat.codes

            # Prediction
            prediction = model.predict(sample)

            # Friendly output
            result_text = "Approved ✅" if prediction[0] == 1 else "Not Approved ❌"

            return f"Result: {result_text}"

        except Exception as e:
            return f"Error: {str(e)}"

# ================= APP =================
app = App(app_ui, server)