# # calculatiing best no. of clusters wih elbow method
# import pandas as pd
# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt
# data = {
#     "CustomerID": list(range(1,31)),
#     "Annual_Income": [15,16,17,18,19,20,21,22,23,24,
#                       25,26,27,28,29,30,31,32,33,34,
#                       35,36,37,38,39,40,41,42,43,44],
#     "Spending_Score": [39,81,6,77,40,76,6,94,3,72,
#                        14,99,15,77,13,79,35,66,29,98,
#                        35,5,73,14,82,32,61,31,87,4]
# }
#
# df=pd.DataFrame(data)
#
# x=df[["Annual_Income","Spending_Score"]]
#
# wiss=[]
#
# for k in range(1,10):
#     model=KMeans(n_clusters=k,random_state=42,n_init=10)
#     model.fit(x)
#     wiss.append(model.inertia_)
#
# plt.plot(range(1,10),wiss,marker="o",color="red")
# plt.xlabel("number of cluster")
# plt.ylabel("within cluster square of sum")
# plt.grid(True)
# plt.show()

# ploting 2d scatter plot for k=4
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
data = {
    "CustomerID": list(range(1,31)),
    "Annual_Income": [15,16,17,18,19,20,21,22,23,24,
                      25,26,27,28,29,30,31,32,33,34,
                      35,36,37,38,39,40,41,42,43,44],
    "Spending_Score": [39,81,6,77,40,76,6,94,3,72,
                       14,99,15,77,13,79,35,66,29,98,
                       35,5,73,14,82,32,61,31,87,4]
}
df=pd.DataFrame(data)

x=df[["Annual_Income","Spending_Score"]]
model=KMeans(n_clusters=4,random_state=42,n_init=10)
df["group"]=model.fit_predict(x)
# print(model)
# print(df)

for group in df["group"].unique():
    new_data=df[df["group"]==group]
    # print(new_data)
    plt.scatter(new_data["Annual_Income"],new_data["Spending_Score"])
plt.xlabel("Annual_Income")

plt.ylabel("Spending_Score")
plt.title("nice work dipesh")
plt.grid(True)
plt.show()


