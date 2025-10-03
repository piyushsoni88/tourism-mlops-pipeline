import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv("data/tourism.csv")

# Data cleaning
df = df.drop(columns=["CustomerID"], errors="ignore")
df = df.fillna(df.mode().iloc[0])

# Train-test split
X = df.drop("ProdTaken", axis=1)
y = df["ProdTaken"]
X = pd.get_dummies(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Save model
joblib.dump(model, "deployment/best_model.pkl")
