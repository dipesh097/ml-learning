import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt ye  h 2d ke liy but inthis we ploting 3d plot

from mpl_toolkits.mplot3d import Axes3D
data = {
    "CustomerID": [1,2,3,4,5,6,7,8,9,10,
                   11,12,13,14,15,16,17,18,19,20,
                   21,22,23,24,25,26,27,28,29,30],
    "Age": [19,21,20,23,31,22,35,23,64,30,
            67,35,58,24,37,22,35,20,52,35,
            25,46,31,54,29,45,35,40,23,60],
    "Annual_Income": [15,15,16,16,17,17,18,18,19,19,
                      19,19,20,20,20,20,21,21,23,23,
                      24,25,25,28,28,28,28,29,29,30],
    "Spending_Score": [39,81,6,77,40,76,6,94,3,72,
                       14,99,15,77,13,79,35,66,29,98,
                       35,5,73,14,82,32,61,31,87,4]
}
df=pd.DataFrame(data)

x=df[["Age","Annual_Income","Spending_Score"]]

model=KMeans(n_clusters=3,random_state=42,n_init=10)

df["group"]=model.fit_predict(x)

fig=plt.figure(figsize=(8,6))
ax=fig.add_subplot(111,projection='3d')

for group in df['group'].unique() :
    group_data=df[df['group']==group]
    ax.scatter(group_data['Age'],group_data['Annual_Income'],group_data['Spending_Score'])

ax.set_xlabel('Age')
ax.set_ylabel('Annual_Income')
ax.set_zlabel('Spending_Score')
plt.title('1st_class_with_big_data')
plt.grid(True)
plt.show()


