import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("data/credit_risk_dataset.csv")

print("\n========== AEGIS CREDIT RISK MODEL ==========\n")

# Features and label
X = data.drop("default", axis=1)
y = data["default"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print(f"Model Accuracy: {round(accuracy*100,2)}%\n")

# Example company prediction
sample_company = X_test.iloc[0].values.reshape(1, -1)

probability = model.predict_proba(sample_company)[0][1]

print("Example Company Risk Assessment:\n")

print(f"Default Probability: {round(probability,2)}")

if probability > 0.6:
    print("⚠ Risk Level: HIGH")
else:
    print("Risk Level: LOW")

print("\n-----------------------------------")