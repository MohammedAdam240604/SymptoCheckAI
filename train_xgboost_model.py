# train_xgboost_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import os

# ✅ 1. Disease ID-to-Name mapping
disease_mapping = {
    0: "Fungal infection", 1: "Allergy", 2: "GERD", 3: "Chronic cholestasis",
    4: "Drug Reaction", 5: "Peptic ulcer disease", 6: "AIDS", 7: "Diabetes",
    8: "Gastroenteritis", 9: "Bronchial Asthma", 10: "Hypertension",
    11: "Migraine", 12: "Cervical spondylosis", 13: "Paralysis (brain hemorrhage)",
    14: "Jaundice", 15: "Malaria", 16: "Chicken pox", 17: "Dengue",
    18: "Typhoid", 19: "Hepatitis A", 20: "Hepatitis B", 21: "Hepatitis C",
    22: "Hepatitis D", 23: "Hepatitis E", 24: "Alcoholic hepatitis",
    25: "Tuberculosis", 26: "Common Cold", 27: "Pneumonia", 28: "Dimorphic hemorrhoids (piles)",
    29: "Heart attack", 30: "Varicose veins", 31: "Hypothyroidism",
    32: "Hyperthyroidism", 33: "Hypoglycemia", 34: "Osteoarthristis",
    35: "Arthritis", 36: "(vertigo) Paroymsal Positional Vertigo",
    37: "Acne", 38: "Urinary tract infection", 39: "Psoriasis",
    40: "Impetigo", 41: "Paralysis (brain hemorrhage)"
}

# ✅ 2. Load and map CSV
data = pd.read_csv("cleaned_symbipredict.csv")
data["prognosis"] = data["prognosis"].map(disease_mapping)

# ✅ 3. Separate features and labels
X = data.drop(columns=["prognosis"])
y = data["prognosis"]

# ✅ 4. Encode labels
disease_encoder = LabelEncoder()
y_encoded = disease_encoder.fit_transform(y)

# ✅ 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ✅ 6. Train model
model = xgb.XGBClassifier(eval_metric="mlogloss")
model.fit(X_train, y_train)

# ✅ 7. Save model and encoder
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
joblib.dump(disease_encoder, "model/disease_encoder.pkl")
joblib.dump(X.columns.tolist(), "model/symptom_vectorizer.pkl")

# ✅ 8. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")

# ✅ 9. Classification report
target_names = disease_encoder.classes_
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))
