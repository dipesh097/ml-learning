import pandas as pd

data={
    "dipesh":[24,45,234,2,1],
    "me":[23,24,1,3,3],
    "you":  [24,4,2,21,34]
}
# df=open.read_csv("adress of file ")                ## for open any file
x=pd.DataFrame(data)
print(x)

print(x.info())


print(x.describe())
