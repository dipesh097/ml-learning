import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# sample data
data = {
    'Hours': [1,2,3,4,5,6,7,8,9,10],
    'Marks': [20,30,35,45,55,65,75,85,90,95]
}

df = pd.DataFrame(data)

# scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# dbscan
dbscan = DBSCAN(eps=0.8, min_samples=3)
clusters = dbscan.fit_predict(X_scaled)

df['Cluster'] = clusters
print(df)
# visialize
plt.scatter(df['Hours'], df['Marks'], c=df['Cluster'])
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.title("DBSCAN Clustering")
plt.show()

# K-distance Graph Method
from sklearn.neighbors import NearestNeighbors
import numpy as np

neighbors = NearestNeighbors(n_neighbors=3)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

distances = np.sort(distances[:, 2])

plt.plot(distances)
plt.xlabel("Points")
plt.ylabel("Distance")
plt.title("K-distance Graph")
plt.show()

