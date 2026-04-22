import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/cleaned_data.csv")

sns.countplot(x='TARGET', data=df)

plt.title("Target Distribution")
plt.xlabel("Target")
plt.ylabel("Count")

plt.show()