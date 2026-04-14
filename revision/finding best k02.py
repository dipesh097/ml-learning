import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


data = {
    'Hours': [1,2,3,4,5,6,7,8,9,10],
    'Marks': [20,30,35,45,55,65,75,85,90,95]
}

df=pd.DataFrame(data)

model=StandardScaler()
scaled_data=model.fit_transform(df)

list=[]

for k in range(1,11):
    kmeans=KMeans(n_clusters=k,random_state=42)
    kmeans.fit(scaled_data)
    list.append(kmeans.inertia_)

plt.plot(range(1,11),list)
plt.show()


model1=KMeans(n_clusters=2,random_state=42)

cluster=model1.fit_predict(scaled_data)

df["cluster"]=cluster

print(df)

plt.scatter(df["Hours"],df['Marks'],c=df["cluster"])
plt.show()