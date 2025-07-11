# app.py

import matplotlib
matplotlib.use('Agg')  # ✅ Prevent GUI/thread issues for pie chart

from flask import Flask, request, render_template, jsonify
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
from utils.predictor import extract_symptoms, vectorize_input
from utils.pdf_generator import generate_pdf, medical_advice
from utils.db_utils import save_prediction, init_db

app = Flask(__name__)

# ✅ Load model and encoders
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/symptom_vectorizer.pkl")
label_encoder = joblib.load("model/disease_encoder.pkl")

# ✅ Clean non-ASCII text (smart quotes, emojis)
def clean_text(text):
    return text.encode('ascii', 'ignore').decode('ascii')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/faq")
def faq():
    return render_template("faq.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        user_input = data.get("user_input", "")
        print(f"User input: {user_input}")

        symptoms = extract_symptoms(user_input)
        print(f"Extracted symptoms: {symptoms}")

        # ✅ No symptom match check
        if not symptoms:
            return jsonify({"error": "No recognizable symptoms found."}), 400

        vec_input = vectorize_input(symptoms, vectorizer)
        prediction_index = model.predict(vec_input)[0]
        predicted_disease = str(label_encoder.inverse_transform([prediction_index])[0])
        print(f"Predicted disease: {predicted_disease}")

        save_prediction(user_input, predicted_disease)

        probabilities = model.predict_proba(vec_input)[0]
        labels = label_encoder.classes_

        top_indices = np.argsort(probabilities)[::-1][:5]
        top_probs = probabilities[top_indices]
        top_labels = labels[top_indices]

        # ✅ Pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(top_probs, labels=top_labels, autopct='%1.1f%%', startangle=140)
        plt.title("Top 5 Disease Probabilities")
        plt.tight_layout()
        chart_path = os.path.join("static", "piechart.png")
        plt.savefig(chart_path)
        plt.close()

        # ✅ Advice
        advice = medical_advice.get(predicted_disease, "Please consult a healthcare provider.")

        # ✅ Generate PDF with cleaned input
        generate_pdf(
            clean_text(user_input),
            clean_text(predicted_disease),
            clean_text(advice),
            {str(labels[i]): float(probabilities[i] * 100) for i in range(len(labels))}
        )

        return jsonify({
            "predicted_disease": predicted_disease,
            "advice": str(advice),
            "symptoms": symptoms,
            "pdf_url": "/static/report.pdf",
            "chart_url": "/static/piechart.png",
            "probabilities": {
                str(label): float(probabilities[i] * 100) for i, label in enumerate(labels)
            }
        })

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

# ✅ Init DB on startup
init_db()

if __name__ == "__main__":
    app.run(debug=True)

