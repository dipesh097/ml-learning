import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score

# Dataset
data = {
    "Word_Count": [120,55,340,78,500,45,600,150,430,90,210,800,65,720,330,270,100,920,45,870,410,160,730,85,560],
    "Link_Count": [5,0,8,1,10,0,12,3,9,1,2,15,0,14,6,4,1,18,0,17,7,2,13,1,11],
    "Special_Char_Count": [10,2,15,3,18,1,20,6,14,3,5,22,1,19,8,7,3,25,1,23,9,4,21,2,16],
    "Spam": [1,0,1,0,1,0,1,0,1,0,0,1,0,1,1,0,0,1,0,1,1,0,1,0,1]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and target
x = df[["Word_Count", "Link_Count", "Special_Char_Count"]]
y = df["Spam"]

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
wc = int(input("Word_Count of mail: "))
lc = int(input("Link_Count of mail: "))
scc = int(input("Special_Char_Count of mail: "))

user = np.array([[wc, lc, scc]])
prediction = model.predict(user)

# Result
if prediction[0] == 1:
    print("📧 This mail is SPAM 🎉👍")
else:
    print("✅ This mail is NOT spam")
