import tkinter as tk
from tkinter import ttk
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("models/model.pkl", "rb"))

# Create main window
root = tk.Tk()
root.title("Credit Card Fraud Prediction")
root.geometry("450x650")

# Title
title = tk.Label(
    root,
    text="Credit Card Fraud Prediction System",
    font=("Arial", 16, "bold")
)
title.pack(pady=10)

# Gender Dropdown
tk.Label(root, text="Gender").pack()
gender = ttk.Combobox(root, values=[0, 1])
gender.pack()

# Car Dropdown
tk.Label(root, text="Car Ownership").pack()
car = ttk.Combobox(root, values=[0, 1])
car.pack()

# Property Dropdown
tk.Label(root, text="Property Ownership").pack()
reality = ttk.Combobox(root, values=[0, 1])
reality.pack()

# Children Entry
tk.Label(root, text="Number of Children").pack()
children = tk.Entry(root)
children.pack()

# Income Entry
tk.Label(root, text="Income").pack()
income = tk.Entry(root)
income.pack()

# Age Entry
tk.Label(root, text="Age").pack()
age = tk.Entry(root)
age.pack()

# Years Employed Entry
tk.Label(root, text="Years Employed").pack()
years = tk.Entry(root)
years.pack()

# Prediction Function
def predict():
    try:
        values = [
            float(gender.get()),
            float(car.get()),
            float(reality.get()),
            float(children.get()),
            float(income.get()),
            float(age.get()),
            float(years.get())
        ]

        prediction = model.predict([values])

        if prediction[0] == 1:
            result = "Approved"
        else:
            result = "Not Approved"

        result_label.config(
            text=f"Prediction Result: {result}"
        )

    except:
        result_label.config(text="Invalid Input")

# Predict Button
predict_btn = tk.Button(
    root,
    text="Predict",
    command=predict,
    bg="blue",
    fg="white"
)
predict_btn.pack(pady=20)

# Result Label
result_label = tk.Label(
    root,
    text="",
    font=("Arial", 14, "bold")
)
result_label.pack()

# Run Application
root.mainloop()