import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score

# Dataset (25 rows)
data = {
    "Size_sqft": [850,900,950,1000,1100,1150,1200,1250,1300,1350,1400,1450,1500,1550,1600,1650,1700,1750,1800,1850,1900,1950,2000,2100,2200],
    "Rooms": [2,2,2,2,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,5,5,5,5,5,5],
    "Age": [1,2,1,3,2,1,4,3,2,1,5,4,3,2,1,6,5,4,3,2,1,7,6,5,4],
    "Distance_city_km": [10,12,9,11,15,14,10,13,12,11,16,14,13,12,11,17,15,14,13,12,11,18,16,15,14],
    "Price_Lakh": [43,47,49,52,56,59,61,64,67,69,72,74,78,81,84,87,90,92,95,99,102,105,108,113,118]
}

df = pd.DataFrame(data)

# Features & target
X = df[["Size_sqft","Rooms","Age","Distance_city_km"]]
y = df["Price_Lakh"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------- Ridge Regression ----------------
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
print("Ridge R² score:", r2_score(y_test, y_pred_ridge))
print("Ridge coefficients:", ridge_model.coef_)

# ---------------- Lasso Regression ----------------
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
print("\nLasso R² score:", r2_score(y_test, y_pred_lasso))
print("Lasso coefficients:", lasso_model.coef_)

# ---------------- ElasticNet Regression ----------------
enet_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet_model.fit(X_train, y_train)
y_pred_enet = enet_model.predict(X_test)
print("\nElasticNet R² score:", r2_score(y_test, y_pred_enet))
print("ElasticNet coefficients:", enet_model.coef_)

# ---------------- User Input Prediction ----------------
print("\nEnter new house details to predict Price:")
size = float(input("Size (sqft): "))
rooms = int(input("Number of rooms: "))
age = int(input("House Age (years): "))
distance = float(input("Distance from city (km): "))

new_house = [[size, rooms, age, distance]]

ridge_price = ridge_model.predict(new_house)[0]
lasso_price = lasso_model.predict(new_house)[0]
enet_price = enet_model.predict(new_house)[0]

print(f"\nPredicted Price (Ridge): {round(ridge_price,2)} Lakh")
print(f"Predicted Price (Lasso): {round(lasso_price,2)} Lakh")
print(f"Predicted Price (ElasticNet): {round(enet_price,2)} Lakh")
