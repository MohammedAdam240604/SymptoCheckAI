import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

# -------------------- 1. Load model and encoder --------------------
model = joblib.load("xgb_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
X_train = pd.read_csv("X_train.csv")
scaler = StandardScaler()
scaler.fit(X_train)

# -------------------- 2. Define input symptoms --------------------
# Start with zeros for all features
input_data = {col: 0 for col in X_train.columns}

# ✅ Manually turn ON the symptoms you want to test
input_data['itching'] = 1
input_data['fatigue'] = 1
input_data['chills'] = 1
input_data['vomiting'] = 1

# -------------------- 3. Create input DataFrame --------------------
input_df = pd.DataFrame([input_data])
input_scaled = scaler.transform(input_df)

# -------------------- 4. Predict disease --------------------
predicted_encoded = model.predict(input_scaled)[0]
predicted_label = label_encoder.inverse_transform([predicted_encoded])[0]

# -------------------- 5. Prediction confidence --------------------
proba = model.predict_proba(input_scaled)[0]
predicted_confidence = proba[predicted_encoded] * 100

# -------------------- 6. Print result --------------------
print("✅ Predicted Disease:", predicted_label)
print(f"🎯 Prediction Confidence: {predicted_confidence:.2f}%")

# -------------------- 7. Show Top 3 Probable Diseases --------------------
top_3 = sorted(enumerate(proba), key=lambda x: x[1], reverse=True)[:3]
print("\n🧠 Top 3 Disease Predictions:")
for idx, score in top_3:
    label = label_encoder.inverse_transform([idx])[0]
    print(f"🔹 {label}: {score*100:.2f}%")

# -------------------- 8. Generate PDF Report --------------------
def generate_pdf_report(symptoms, disease, confidence, filename="prediction_report.pdf"):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 50, "🩺 SymptoCheck - Disease Prediction Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(50, height - 130, f"Symptoms Provided:")
    c.drawString(70, height - 150, ", ".join(symptoms))
    c.drawString(50, height - 190, f"Predicted Disease: {disease}")
    c.drawString(50, height - 220, f"Prediction Confidence: {confidence:.2f}%")

    c.save()
    print(f"\n📄 PDF report saved as '{filename}'")

# Extract symptoms marked as 1
symptoms_used = [k for k, v in input_data.items() if v == 1]

# Call the PDF function
generate_pdf_report(symptoms_used, predicted_label, predicted_confidence)
