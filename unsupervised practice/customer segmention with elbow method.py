import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {
    "CustomerID": [1,2,3,4,5,6,7,8,9,10],
    "Age": [19,21,20,23,31,22,35,23,64,30],
    "Annual_Income": [15,15,16,16,17,17,18,18,19,19],
    "Spending_Score": [39,81,6,77,40,76,6,94,3,72]
}
df=pd.DataFrame(data)
x=df[["Annual_Income","Spending_Score"]]

inertia=[]

for i in range(1,11):
    model=KMeans(n_clusters=i,random_state=42)
    model.fit(x)
    inertia.append(model.inertia_)

plt.scatter(range(1,11),inertia,marker="*")
plt.title("finding k with elbow method ")
plt.xlabel("no. of cluster")
plt.ylabel("inertia(distance b/w center and point)")
plt.grid(True)
# plt.show()

# here best cluster no. is 2 , according elbow method
model1=KMeans(n_clusters=2,random_state=43)
df["cluster"]=model1.fit_predict(x)

plt.scatter(df["Annual_Income"],df[ "Spending_Score"],c=df["cluster"],cmap="rainbow")
plt.title("clustring with 2 clusters")
plt.xlabel("Annual_Income")
plt.ylabel("Spending_Score")
plt.grid(True)
plt.show()

