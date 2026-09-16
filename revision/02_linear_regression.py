import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


data = {
    'hours': [
        0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5,
        5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5,
        10, 1.2, 2.8, 3.7, 4.9, 5.3, 6.1, 7.2, 8.4, 9.1,
        2.2, 3.1, 4.2, 5.8, 6.9, 7.7, 8.8, 9.6, 1.8, 2.6,
        3.9, 4.7, 5.1, 6.4, 7.9, 8.1, 9.3, 10.5, 11, 12
    ],
    'marks': [
        0, 5, 10, 18, 22, 28, 35, 42, 50, 55,
        60, 64, 70, 74, 78, 82, 85, 88, 90, 92,
        95, 14, 30, 40, 54, 62, 71, 79, 87, 91,
        25, 36, 48, 67, 76, 83, 89, 93, 20, 32,
        45, 52, 58, 73, 84, 86, 92, 97, 98, 100
    ]
}

df=pd.DataFrame(data)

x=df[['hours']]
y=df['marks']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

model=LinearRegression()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)
user=int(input("your average study hours :"))
marks=model.predict([[user]])

print("r2 score of this model is :",r2_score(y_test,y_pred))

plt.scatter(x,y)
plt.plot(x,model.predict(x))
plt.show()