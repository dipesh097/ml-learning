import numpy as np
import pandas as pd

data={
    "dipesh":[24,45,234,2,np.nan],
    "me":[23,24,np.nan,3,3],
    "you":  [24,np.nan,2,21,34]
}

df=pd.DataFrame(data)

print(df.isnull().sum())

df.fillna(df.mean()  ,inplace=True)

print(df)