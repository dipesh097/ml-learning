from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris=load_iris()
x=iris.data
y=iris.target


x_train , x_test, y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)




scaler=StandardScaler()

x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)


svt=SVC(kernel="linear")
svt.fit(x_train_scaled,y_train)

print("svm accuracy :",svt.score(x_test_scaled,y_test))