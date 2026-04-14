import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

# Dataset

data = {
    "Age": [25,30,35,40,45,50,55,60,65,70,28,33,38,43,48,53,58,63,68,73,27,32,37,42,47],
    "Blood_Pressure": [120,122,125,130,135,140,145,150,155,160,121,124,128,133,138,142,148,152,158,162,119,123,127,131,137],
    "Cholesterol": [180,185,190,200,210,220,230,240,250,260,182,188,195,205,215,225,235,245,255,265,178,184,192,202,212],
    "Heart_Rate": [70,72,74,75,78,80,82,84,86,88,71,73,76,79,81,83,85,87,89,90,69,71,74,77,79],
    "Disease": [0,0,0,0,1,1,1,1,1,1,0,0,0,1,1,1,1,1,1,1,0,0,0,1,1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
x = df[["Age","Blood_Pressure","Cholesterol","Heart_Rate"]]
y = df["Disease"]

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(x_train, y_train)

# Predictions and evaluation
y_pred = model.predict(x_test)

print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Precision Score:", precision_score(y_test, y_pred))

# User input
Age = int(input("enter  age : "))
Blood_Pressure = int(input("Blood_Pressure :"))
Cholesterol= int(input("Cholesterol :"))
Heart_Rate= int(input("Heart_Rate :"))

user = np.array([[Age,Blood_Pressure,Cholesterol,Heart_Rate]])
prediction = model.predict(user)

# cofidance showing (probability)
prob=model.predict_proba(user)[0][0]*100
print(f"chanses for not disease : {prob:.2f}%")
# Result
if prediction[0] == 1:
    print("disease ")
else:
    print("not disease ")
