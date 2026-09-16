import pandas as pd
import numpy as np

df = pd.DataFrame({
    "Name": ["Amit", "Sara", None, "Riya"],
    "Age": [21, np.nan, 20, 23],
    "Score": [88, 95, np.nan, 91]
})

print(df)
print(df.isnull)
print(df.isnull().sum())
print(df.dropna())

print(df.fillna(1))