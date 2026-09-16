import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = {
    "Car_Age": [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
    "Mileage_kmpl": [22,21.5,21,20.3,19.5,18.9,18.1,17.6,17.2,16.8,16.5,16.2,16.1,16.0,15.9,15.8,15.7,15.6,15.5,15.4,15.3,15.2,15.2,15.1,15.0]
}
df=pd.DataFrame(data)

x=df[["Car_Age"]]
y=df[[ "Mileage_kmpl"]]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model1=PolynomialFeatures(degree=2)
x_train_poly=model1.fit_transform(x_train)
x_test_poly=model1.transform(x_test)

model2=LinearRegression()
model2.fit(x_train_poly,y_train)

y_pred=model2.predict(x_test_poly)
print("r2 value is :",r2_score(y_test,y_pred))

user=int(input("enter your car's age :"))

user_poly=model1.transform([[user]])
prediction=model2.predict(user_poly)

print("expected Mileage_kmpl is ", prediction[0])


