import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path

# Load dataset
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Fraudulent_E_Commerce_Transactions.csv"
df = pd.read_csv(DATA_PATH)

# Select features and target
label_col = "Is Fraudulent"
feature_cols = [
    "Transaction Amount",
    "Quantity",
    "Customer Age",
    "Account Age Days",
    "Transaction Hour",
]

X = df[feature_cols]
y = df[label_col]

# Encode target if necessary
if y.dtype == object:
    le = LabelEncoder()
    y = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Baseline Model Accuracy:", round(acc, 4))

# Save trained model
joblib.dump(model, "baseline_fraud_model.joblib")
print("🚀 Model saved as baseline_fraud_model.joblib")
