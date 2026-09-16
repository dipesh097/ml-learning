import pandas as pd

data=pd.DataFrame({
        "name": ["amit", "sara", "john"],
        "score":[97,95,78],
         "age":[21,22,20]

})

print(data["name"])
print(data[["name"]])

print(data[["name","score"]])

# Selecting Rows by Position — iloc

print(data.iloc[0])
print(data.iloc[2])
print(data.loc[1,"name"])
print(data.iloc[1,0])

# i have to print johns score by using both

print(data.loc[2,"score"])
print(data.iloc[2,2])

# we have to print  age of amit and sara

print(data.loc[0:1,["name","age"]])
print(data.iloc[0:2,0:2])

# ---------------------------------sorting data-----------------------------------------------

print(data.sort_values("age"))
print(data.sort_values("age",ascending=False))

print(data.sort_values(["score","age"]))

print(data.sort_values("score",inplace=False))
print(data.sort_values("score",inplace=True))


