import pandas as pd
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
data = {
    "Annual_Income": [25,27,29,26,28,     # Cluster 1 (low income, low spending)
                      60,62,65,63,61,     # Cluster 2 (mid income, mid spending)
                      85,88,86,89,90,     # Cluster 3 (high income, high spending)
                      40,42,43,41,39],    # Cluster 4 (mid income, low spending)

    "Spending_Score": [20,22,19,24,21,    # Cluster 1
                       55,58,53,56,54,    # Cluster 2
                       90,92,88,95,91,    # Cluster 3
                       30,33,28,31,29]    # Cluster 4
}
df=pd.DataFrame(data)

x=df[[  "Annual_Income","Spending_Score"]]

dend=sch.dendrogram(sch.linkage(x,method="ward"))

plt.grid(True)
plt.show()


# use in this agglomerativ clustring
from sklearn.cluster import AgglomerativeClustering

model =AgglomerativeClustering(n_clusters=4,linkage="ward")
df["group"]=model.fit_predict(x)

print(df)