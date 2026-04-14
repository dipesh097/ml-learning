import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {
    "House_ID": [1,2,3,4,5,6,7,8,9,10],
    "Area": [1200,1500,800,2000,1800,900,2500,3000,1600,1100],
    "Bedrooms": [2,3,2,4,3,2,4,5,3,2],
    "Price": [35,50,25,75,65,30,85,120,55,32]
}

df=pd.DataFrame(data)

x=df[["Area","Bedrooms","Price"]]

inertia=[]
for i in range(1,11):
    model=KMeans(n_clusters=i,random_state=42)
    model.fit(x)
    inertia.append(model.inertia_)
plt.scatter(range(1,11),inertia)
# plt.show()

#  so according to elbow method best no. of clusters is 3


# because there is coming many outlier without scaling , so we need to scale , you can see by commenting scaler process
from sklearn.preprocessing import StandardScaler

model2=StandardScaler()
x_scaled=model2.fit_transform(x)

model1=KMeans(n_clusters=3,random_state=42)
df["cluster"]=model1.fit_predict(x_scaled)
# print(df)

# plt.scatter(df["Area"],df[ "Price"],c=df["cluster"],cmap="rainbow")
plt.scatter(x_scaled[:,0],x_scaled[:,2],c=df["cluster"],cmap="rainbow",s=100)
plt.title("house price clustring")
plt.xlabel("independent variables")
plt.ylabel("dependent variable")
plt.grid(True)
plt.show()