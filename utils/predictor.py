# utils/predictor.py

import joblib
import numpy as np
import re
from fuzzywuzzy import fuzz

# ✅ Load model components
model = joblib.load("model/model.pkl")
disease_encoder = joblib.load("model/disease_encoder.pkl")
feature_list = joblib.load("model/symptom_vectorizer.pkl")

# ✅ Symptom Synonym Map (Extended)
synonym_map = {
    "abdominal pain": "abdominal_pain",
    "belly pain": "abdominal_pain",
    "pain in stomach": "abdominal_pain",
    "stomach ache": "abdominal_pain",
    "bloating": "abdominal_pain",
    "vomiting": "vomiting",
    "throwing up": "vomiting",
    "nauseous": "nausea",
    "feeling sick": "nausea",
    "tired": "fatigue",
    "weak": "fatigue",
    "super weak": "fatigue",
    "dizzy": "dizziness",
    "spinning": "dizziness",
    "lightheaded": "dizziness",
    "head is spinning": "dizziness",
    "sore throat": "sore_throat",
    "itchy": "itching",
    "skin itchy": "itching",
    "rashes": "skin_rash",
    "rash": "skin_rash",
    "skin rash": "skin_rash",
    "red spots": "skin_rash",
    "discoloration": "dischromic_patches",
    "discolored skin": "dischromic_patches",
    "chest hurts": "chest_pain",
    "chest pain": "chest_pain",
    "tight chest": "chest_pain",
    "shortness of breath": "breathlessness",
    "trouble breathing": "breathlessness",
    "difficulty breathing": "breathlessness",
    "can’t breathe": "breathlessness",
    "burning while urinating": "burning_micturition",
    "pain while peeing": "burning_micturition",
    "frequent urination": "polyuria",
    "peeing often": "polyuria",
    "urine smells bad": "foul_smell_of_urine",
    "body hot": "fever",
    "feverish": "fever",
    "high temperature": "fever",
    "hot body": "fever",
    "dry cough": "cough",
    "wet cough": "cough",
    "can’t stop coughing": "cough",
    "lost smell": "loss_of_smell",
    "lost taste": "loss_of_taste",
    "sensitive to light": "photophobia",
    "sensitive to sound": "phonophobia",
    "neck pain": "pain_in_neck",
    "back pain": "pain_in_lower_back",
    "sweaty": "sweating",
    "lots of sweat": "sweating",
    "feeling hot": "fever",
    "cold": "chills",
    "shivering": "chills",
    "runny nose": "runny_nose",
    "stuffy nose": "congestion",
    "can’t sleep": "lack_of_sleep",
    "head hurts": "headache",
    "head pain": "headache",
    "sick feeling": "malaise",
    "can’t eat": "loss_of_appetite",
    "no appetite": "loss_of_appetite",
    "low appetite": "loss_of_appetite",
    "dark urine": "dark_urine",
    "yellow eyes": "yellowing_of_eyes",
    "yellow skin": "yellowish_skin"
}

def clean_and_split(text):
    """
    Split input into sentences/chunks to allow better matching.
    """
    text = text.lower()
    return re.split(r'[.,!?;🧑]', text)

def extract_symptoms(user_input):
    """
    Extracts symptoms using synonym mapping and fuzzy matching.
    Handles flexible, real-world, even unstructured user inputs.
    """
    user_input = user_input.lower()
    chunks = clean_and_split(user_input)
    matched = set()

    # ✅ Match based on synonym phrases
    for phrase, mapped_symptom in synonym_map.items():
        if phrase in user_input:
            matched.add(mapped_symptom)

    # ✅ Fuzzy match from known symptom list
    for sentence in chunks:
        for symptom in feature_list:
            clean_symptom = symptom.replace("_", " ").strip()
            if clean_symptom in sentence:
                matched.add(symptom)
            else:
                score = fuzz.partial_ratio(clean_symptom, sentence)
                if score >= 80:
                    matched.add(symptom)

    return list(matched)

def vectorize_input(symptoms_list, vectorizer=None):
    """
    Converts extracted symptoms into a binary input vector.
    """
    features = vectorizer if vectorizer else feature_list
    input_vector = np.zeros(len(features), dtype=int)

    for symptom in symptoms_list:
        if symptom in features:
            index = features.index(symptom)
            input_vector[index] = 1

    return input_vector.reshape(1, -1)

def predict_disease(symptoms_list):
    """
    Predicts disease from a list of symptoms.
    """
    input_vector = vectorize_input(symptoms_list)
    prediction_index = model.predict(input_vector)[0]
    return disease_encoder.inverse_transform([prediction_index])[0]
