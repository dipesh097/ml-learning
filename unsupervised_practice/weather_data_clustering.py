import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = {
    "Day": [1,2,3,4,5,6,7,8,9,10],
    "Temperature": [30,32,35,25,28,20,18,15,10,22],
    "Humidity": [65,60,55,80,70,85,90,95,88,75]
}

df = pd.DataFrame(data)

x=df[["Temperature","Humidity"]]

inertia=[]

for i in range(1,11):
    model=KMeans(n_clusters=i,random_state=42)
    model.fit(x)
    inertia.append(model.inertia_)
# print(inertia)

plt.scatter(range(1,11),inertia)
# plt.show()
# here 3 cluster points are best
# so without scaleing there is coming some outliers , so we scale data
from sklearn.preprocessing import StandardScaler
model2=StandardScaler()
x_scaled=model2.fit_transform(x)
model1=KMeans(n_clusters=3,random_state=42)
df["cluster"]=model1.fit_predict(x_scaled)

plt.scatter(x_scaled[:,0],x_scaled[:,1],c=df["cluster"],marker="o")
plt.title("wether clutring with elbow method")
plt.xlabel("Temperature")
plt.ylabel("Humidity")
plt.grid(True)
plt.show()