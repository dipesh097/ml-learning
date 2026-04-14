
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [30, 35, 45, 50, 55, 60, 65, 70, 75, 80]
}

df=pd.DataFrame(data)
x=df[["Hours"]]
y=df["Marks"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("r2 score is :",r2_score(y_test,y_pred))

user=int(input("enter your study hours:"))

prediction=model.predict([[user]])

print("your expacted marks is:",prediction[0])

plt.plot(x,y)
plt.title("marks prediction")
plt.grid(True)




