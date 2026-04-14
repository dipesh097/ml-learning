import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data ={
    "Day": list(range(1, 26)),
    "Humidity": [88,84,82,80,78,77,75,74,73,72,70,69,68,67,66,65,64,63,62,61,60,59,58,57,56],
    "WindSpeed": [12,14,11,13,10,9,8,7,8,9,10,12,11,10,9,8,7,8,9,10,12,13,14,15,16],
    "Pressure": [1012,1013,1012,1014,1013,1015,1014,1015,1016,1016,1017,1018,1017,1018,1019,1020,1019,1021,1022,1021,1022,1023,1024,1023,1025],
    "Temperature": [28.5,28.3,28.7,29.1,29.4,29.8,30.0,30.2,30.5,30.7,31.0,31.2,31.4,31.6,31.8,32.0,32.2,32.4,32.6,32.8,33.0,33.1,33.2,33.3,33.5]
}
df=pd.DataFrame(data)
x=df[["Day","Humidity","WindSpeed","Pressure"]]
y=df[[ "Temperature"]]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("r2 score is :",r2_score(y_test,y_pred))

day=int(input("which day you want to temprature :"))
Humidity=int(input("how much humidity :"))
WindSpeed=int(input("how much WindSpeed:"))
Pressure=int(input("how much Pressure :"))

new_data=[day,Humidity,WindSpeed,Pressure]

prediction=model.predict([new_data])

print("so expactected temprature is around ",prediction[0])

plt.plot(df["Day"],df["Temperature"])
plt.title("temprature forcasting")
plt.grid(True)
plt.show()
