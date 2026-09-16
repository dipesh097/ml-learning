import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import  matplotlib.pyplot as plt
import numpy as np


data = {
    'Hours': [2, 4, 6, 8, 10, 3, 7, 9],
    'Attendance': [60, 70, 80, 90, 95, 65, 85, 92],
    'Marks': [35, 50, 65, 80, 90, 40, 75, 85]
}

df = pd.DataFrame(data)

x=df[['Hours','Attendance']]
y=df['Marks']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)


hour=int(input("enter your study hour:"))
attendance=int(input("enter your attendance :"))

user_input=np.array([[hour,attendance]])

marks=model.predict(user_input)

print("marks are :",marks)

y_pred=model.predict(x_test)
print("r2 score is :",r2_score(y_test,y_pred))
#
# plt.scatter(x,y)
# plt.plot(x,model.predict(x))
# plt.show()
plt.scatter(y_test,y_pred)
plt.xlabel("actual values")
plt.ylabel("predicted values")
plt.show()
print("cofficients",model.coef_)
print("model inter.",model.intercept_)

