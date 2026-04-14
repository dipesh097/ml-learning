import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = {
    'Hours': [
        0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5,
        5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5,
        10, 10.5, 11, 11.5, 12,
        1.2, 1.8, 2.3, 2.9, 3.4, 3.9, 4.2, 4.8,
        5.1, 5.6, 6.2, 6.8, 7.3, 7.9, 8.4, 8.9,
        9.4, 9.8, 10.2, 10.7, 11.2, 11.7,
        2.1, 3.1, 4.1, 5.9, 6.9, 7.8, 8.6, 9.7
    ],
    'Result': [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 1, 0,
        0, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 1, 1
    ]
}

df=pd.DataFrame(data)

x=df[['Hours']]
y=df['Result']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=StandardScaler()
x_train_scaled=model.fit_transform(x_train)
x_test_scaled=model.transform(x_test)

model1=KNeighborsClassifier(n_neighbors=7,metric='euclidean')
model1.fit(x_train_scaled,y_train)

y_pred=model1.predict(x_test_scaled)

print("accuracy score :",accuracy_score(y_test,y_pred))
print("confusion metrics :",confusion_matrix(y_test,y_pred))


user=float(input("enter your study hours :"))

user_scaled=model.transform([[user]])
result=model1.predict(user_scaled)

if result[0]==1:
    print("🎉congrats, you are passed")
else:
    print("😔 you are failed")
for k in range(1, 11):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train_scaled, y_train)
    acc = accuracy_score(y_test, knn.predict(x_test_scaled))
    print(f"K={k}, Accuracy={acc}")


