# utils/db_utils.py

import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        user_input TEXT,
        predicted_disease TEXT
    )
    """)
    conn.commit()
    conn.close()

def save_prediction(user_input, predicted_disease):
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO predictions (timestamp, user_input, predicted_disease) VALUES (?, ?, ?)", (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        user_input,
        predicted_disease
    ))
    conn.commit()
    conn.close()
