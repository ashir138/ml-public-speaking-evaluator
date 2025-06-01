import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
df = pd.read_csv("extracted_features.csv")

# 2. Prepare features and labels
X = df.drop(columns=['file', 'emotion'])  # numeric features only
y = df['emotion']                         # target class

# 3. Encode labels to numbers
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# 4. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 5. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Predict & Evaluate
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {acc:.2f}\n")
print("📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 7. Confusion Matrix
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# 8. Save model
import joblib
joblib.dump(model, "ml_training/emotion_classifier.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
print("✅ Model and encoder saved.")

