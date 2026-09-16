import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    "Annual_Income": [15,16,17,18,19,20,25,26,27,28,29,30,35,36,37,38,39,40,45,46,47,48,49,50,55],
    "Spending_Score": [80,78,75,73,70,68,60,58,55,52,50,48,40,38,35,32,30,28,25,22,20,18,15,12,10]
}

df=pd.DataFrame(data)

inertia=[]

for i in range(1,11):
    model=KMeans(n_clusters=i,random_state=42)
    model.fit(df)
    inertia.append(model.inertia_)
print(inertia)

plt.scatter(range(1,11),inertia,marker="8")
plt.xlabel("no. of clusters ")
plt.ylabel("inertia")
plt.title("finding no. of clusters by elbow method ")
plt.grid(True)
plt.show()
