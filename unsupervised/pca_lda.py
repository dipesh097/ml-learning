from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
import matplotlib.pyplot as plt

iris=load_iris()
x=iris.data
y=iris.target

Scale=StandardScaler()

x_scaled=Scale.fit_transform(x)
# -------- with PCA -----------
pca=PCA(n_components=2)

pca_data=pca.fit_transform(x_scaled)

plt.scatter(pca_data[:,0],pca_data[:,1])

plt.xlabel("pca component 1st")
plt.ylabel("pca component 2nd")
plt.title("analysis with pca")
plt.grid(True)
plt.show()
# ------- with lda--------
Lda=lda(n_components=2)
lda_data=Lda.fit_transform(x_scaled,y)
plt.scatter(lda_data[:,0],lda_data[:,1],color="blue")
print(lda_data)


plt.xlabel("lda component 1st")
plt.ylabel("lda component 2nd")
plt.title("analysis with lda")
plt.grid(True)
plt.show()
