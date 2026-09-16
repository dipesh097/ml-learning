import pandas as pd
from sklearn.cluster import AgglomerativeClustering, ward_tree
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

data = {
    'Hours': [1,2,3,4,5,6,7,8,9,10],
    'Marks': [20,30,35,45,55,65,75,85,90,95]
}

df=pd.DataFrame(data)

model1=StandardScaler()
scaled_data=model1.fit_transform(df)

model=AgglomerativeClustering(n_clusters=2,linkage="ward")

df["cluster"]=model.fit_predict(scaled_data)
print(df)

plt.scatter(df['Hours'],df['Marks'],c=df["cluster"])
plt.show()

#Code to plot dendrogram
import scipy.cluster.hierarchy as sch

plt.figure(figsize=(8,5))
sch.dendrogram(sch.linkage(scaled_data, method='ward'))
plt.title("Dendrogram")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()

