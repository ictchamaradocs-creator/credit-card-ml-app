# Credit Card Approval Prediction System

## Overview

This project is a machine learning application that predicts whether a credit card application will be approved or not. It is built using Python and Shiny.

## Features

* User input: Age, Income, Family Size, Gender, Car Ownership, House Ownership
* Output: Approval result and confidence score
* Basic data visualizations (age, income, approval distribution)

## Technologies

* Python
* Pandas, Scikit-learn
* Matplotlib
* Shiny (Python)

## How to Run

```bash
git clone https://github.com/ictchamaradocs-creator/credit-card-ml-app.git
cd credit-card-ml-app
pip install -r requirements.txt
python -m shiny run app.py
```

Open in browser: http://127.0.0.1:8000

## Notes

* The model uses multiple features, but only key inputs are exposed in the UI
* High accuracy may indicate overfitting
* Data quality affects prediction results

## Author

Inosha Wijesinghe
