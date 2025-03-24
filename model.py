import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv("water_quality.csv")
print(df.isnull().sum())

df.fillna(df.mean(), inplace=True)

X = df.drop(columns=["Potability"])
y = df["Potability"]

y = pd.to_numeric(y, errors='coerce')

df.dropna(subset=["Potability"], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if np.any(np.isinf(X_train)) or np.any(np.isinf(y_train)):
    print("Dataset contains infinite values. Please clean the dataset.")
    exit()

model = SVC(kernel="rbf")
model.fit(X_train, y_train)

with open("svm_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")