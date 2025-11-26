import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
import joblib

# Load dataset
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Fraudulent_E_Commerce_Transactions.csv"
df = pd.read_csv(DATA_PATH)

# Target and features
target = "Is Fraudulent"
features = [
    "Transaction Amount",
    "Payment Method",
    "Product Category",
    "Quantity",
    "Customer Age",
    "Customer Location",
    "Device Used",
    "Account Age Days",
    "Transaction Hour"
]

X = df[features]
y = df[target]

# Column types
numeric_features = [
    "Transaction Amount",
    "Quantity",
    "Customer Age",
    "Account Age Days",
    "Transaction Hour"
]

categorical_features = [
    "Payment Method",
    "Product Category",
    "Customer Location",
    "Device Used"
]

# Preprocessing pipeline
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# Model
model = GradientBoostingClassifier()

# Full pipeline
pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("model", model)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

# Evaluation
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_proba)
print("ROC AUC Score:", round(auc, 3))

# Save trained model
joblib.dump(pipeline, "advanced_fraud_model.joblib")
print("\n🚀 Advanced model saved as advanced_fraud_model.joblib")


Add advanced ML model pipeline with preprocessing and ROC AUC metrics

