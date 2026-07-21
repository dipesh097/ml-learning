import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data={
    "Size_sqft": [850,900,950,1000,1100,1150,1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000,2100,2200],
    "Rooms": [2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5],
    "Price_Lakh": [43,47,49,52,56,59,61,64,67,69,72,74,78,81,84,87,90,92,95,99,102,105,108,113,118]
}

df=pd.DataFrame(data)

x=df[["Size_sqft","Rooms"]]
y=df[["Price_Lakh"]]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("r2 score is ",r2_score(y_test,y_pred))
# print("accuracy_score is ",accuracy_score(y_test,y_pred))

size=int(input("enter size of your house(sqft):"))
Rooms=int(input("how much rooms in  your house(sqft):"))

new_data=[size,Rooms]

prediction=model.predict([new_data])

print("the expacted price of your house is :",prediction[0])
