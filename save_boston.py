# save_boston.py
from sklearn.datasets import fetch_openml
import pandas as pd

# Fetch dataset
boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame

# Save to CSV
df.to_csv("data/raw_data.csv", index=False)
print("Saved data/raw_data.csv")
