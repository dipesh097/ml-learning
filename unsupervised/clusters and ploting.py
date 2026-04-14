import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data={
   "custmoer":["dipesh","tushar","manish","sachin"],
    "age" :[18,57,30,26],
    "spending":[100,380,300,200]
}
df=pd.DataFrame(data)

x=df[["age","spending"]]

model=KMeans(n_clusters=2,random_state=42,n_init=10)

df['group'] =model.fit_predict(x)

plt.figure(figsize=(6,5))

for group in df['group'].unique():
    group_data=df[df['group']==group]
    plt.scatter(group_data['age'],group_data['spending'],label=f'group{group}')
plt.xlabel('age')
plt.ylabel('spending')
plt.title('customer segments(k-means)')
plt.legend()
plt.grid(True)
plt.show()
print(df)




