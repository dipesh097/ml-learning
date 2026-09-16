
# Write code for:
#
# Students with Score > 90
# Students with Age < 22
# Students with Age > 20 AND Score > 85
# Students whose Name is "John"
# Students whose Score is NOT greater than 90
import pandas as pd

df = pd.DataFrame({
    "Name": ["Amit", "Sara", "John", "Riya"],
    "Age": [21, 22, 20, 23],
    "Score": [88, 95, 78, 91]
})

a=df[df["Score"]>90]
print(a)

b=df[df["Age"]<20]
print(b)

c=df[(df["Age"]>20) & (df["Score"]>85)]
print(c)

d=df[df["Name"]=="John"]
print(d)

e=df[df["Score"]<90]
print(e)


