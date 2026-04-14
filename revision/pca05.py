import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# sample data
data = {
    'Hours': [2,4,6,8,10,3,7,9],
    'Attendance': [60,70,80,90,95,65,85,92],
    'Marks': [35,50,65,80,90,40,75,85]
}

df = pd.DataFrame(data)

X = df[['Hours', 'Attendance', 'Marks']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(X_pca)

# 🔹 9️⃣ PCA Visualization (2D)
plt.scatter(X_pca[:,0], X_pca[:,1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Projection")
plt.show()
