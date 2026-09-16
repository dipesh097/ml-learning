# Create this DataFrame:
#
# Name	Age	Score
# Amit	21	88
# Sara	22	95
# John	20	78
#
# Then do:
#
# Print first 2 rows
# Print shape
# Print column names
# Run describe()

import pandas as pd

data=pd.DataFrame({
        "name": ["amit", "sara", "john"],
        "age":[21,22,20],
        "score":[88,95,78]
})

# print(data)
# print(data.head(2))
# print(data.columns)
# print(data.describe())
# # print(data.tail(2))
# data.info()

