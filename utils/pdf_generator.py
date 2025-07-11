# utils/pdf_generator.py

from fpdf import FPDF
import os

# ✅ Disease-specific advice
medical_advice = {
    "Typhoid": "Take antibiotics as prescribed and stay hydrated.",
    "Common Cold": "Rest, drink fluids, and use OTC medications if needed.",
    "Malaria": "Use antimalarial medication and avoid mosquito bites.",
    "Fungal infection": "Use antifungal creams and maintain hygiene.",
    "Heart attack": "Seek emergency medical attention immediately.",
    "Dengue": "Rest, take paracetamol, and avoid aspirin. Drink fluids.",
    "Jaundice": "Avoid alcohol, take prescribed meds, and rest your liver.",
    "Hyperthyroidism": "Take antithyroid medication and monitor hormone levels.",
    "Chronic cholestasis": "Follow low-fat diet, vitamin supplements, and medication.",
    "Urinary tract infection": "Use antibiotics and drink plenty of water."
}

# ✅ PDF generator (cleaned input expected)
def generate_pdf(user_input, predicted_disease, advice, probability_dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # ✅ Title — removed emoji for safe encoding
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Symptom Analysis Report", ln=True, align='C')
    pdf.ln(10)

    # ✅ Report Details
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, f"User Symptoms: {user_input}")
    pdf.ln(2)
    pdf.multi_cell(0, 10, f"Predicted Disease: {predicted_disease}")
    pdf.ln(2)
    pdf.multi_cell(0, 10, f"Medical Advice: {advice}")
    pdf.ln(10)

    # ✅ Top Predictions
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Prediction Probabilities:", ln=True)
    pdf.set_font("Arial", size=11)

    top_5 = sorted(probability_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    for label, prob in top_5:
        line = f"{label}: {round(prob, 2)}%"
        pdf.cell(0, 10, line, ln=True)

    # ✅ Pie chart
    chart_path = os.path.join("static", "piechart.png")
    if os.path.exists(chart_path):
        pdf.image(chart_path, x=40, y=pdf.get_y() + 10, w=130)

    # ✅ Save final PDF
    pdf.output("static/report.pdf")
