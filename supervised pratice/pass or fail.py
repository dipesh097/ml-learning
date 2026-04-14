import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score , precision_score
data = {
    "Hours_Studied": [1,2,3,4,5,1.5,2.5,3.5,4.5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,11,12,13,14,15,16],
    "Passed": [0,0,0,0,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
}

df=pd.DataFrame(data)

x=df[[ "Hours_Studied"]]
y=df["Passed"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("accuracy score is :",accuracy_score(y_test,y_pred))
print("precision score  is :",precision_score(y_test,y_pred))

user=float(input("enter you study hours :"))

prediction=model.predict([[user]])

if prediction==1 :
    print("congrats you are passed 🎉👍")
else:
    print("sorry , you are failed 😒")