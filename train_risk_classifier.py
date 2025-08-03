import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# ========== Step 1: Load Clustered Customer Data ==========
input_path = "outputs/customer_risk_clusters.csv"
df = pd.read_csv(input_path)

# ========== Step 2: Encode Risk Levels ==========
le = LabelEncoder()
df['risk_label'] = le.fit_transform(df['risk_level'])  # E.g., Low Risk = 1, Medium = 2, High = 0

# ========== Step 3: Define Features and Target ==========
features = [
    'total_invoices', 'avg_days_late', 'max_days_late',
    'total_days_late', 'avg_invoice_amount', 'total_disputes'
]
X = df[features]
y = df['risk_label']

# ========== Step 4: Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ========== Step 5: Train Classifier ==========
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# ========== Step 6: Evaluate Model ==========
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

print(f"✅ Accuracy: {accuracy:.4f}")
print("📊 Classification Report:")
print(report)

# ========== Step 7: Save Model and Encoder ==========
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/risk_classifier.pkl")
joblib.dump(le, "models/risk_label_encoder.pkl")

print("\n✅ Model saved as: models/risk_classifier.pkl")
print("✅ Label encoder saved as: models/risk_label_encoder.pkl")
