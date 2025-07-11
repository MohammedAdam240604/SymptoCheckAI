import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

# -------------------- 1. Load CSV --------------------
file_path = "symbipredict_2022.csv"
df = pd.read_csv(file_path)

print("\n🔹 Original Data:")
print(df.shape)

# -------------------- 2. Drop duplicates & missing --------------------
df.drop_duplicates(inplace=True)
df.dropna(thresh=int(0.4 * len(df)), axis=1, inplace=True)
df.ffill(inplace=True)
df.bfill(inplace=True)

# -------------------- 3. Encode categorical --------------------
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = df[col].astype(str)
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    print(f"✅ Encoded: {col}")

    # ✅ Save label encoder for target column
    if col == 'prognosis':
        joblib.dump(le, "label_encoder.pkl")
        print("💾 Saved label encoder to label_encoder.pkl")

# -------------------- 4. Remove low variance & high correlation --------------------
low_var_cols = [col for col in df.columns if df[col].nunique() <= 1]
df.drop(columns=low_var_cols, inplace=True)

corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
high_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
df.drop(columns=high_corr, inplace=True)

# -------------------- 5. Feature engineering --------------------
if 'Age' in df.columns and 'Cholesterol' in df.columns:
    df['Age_Cholesterol_Interaction'] = df['Age'] * df['Cholesterol']
    print("➕ Created: Age_Cholesterol_Interaction")

# -------------------- 6. Split features and target --------------------
target_col = 'prognosis'
X = df.drop(columns=[target_col])
y = df[target_col]

# -------------------- 7. Train-Test Split --------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n📊 Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# -------------------- 8. Feature Scaling --------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------- 9. Safe SMOTE --------------------
class_counts = Counter(y_train)
min_class_count = min(class_counts.values())
k_neighbors = min(5, min_class_count - 1)

if k_neighbors < 1:
    raise ValueError("❌ Not enough samples in one of the classes to apply SMOTE.")

print(f"🔁 Applying SMOTE with k_neighbors={k_neighbors}")
smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
print("✅ SMOTE applied. Balanced training shape:", X_train_balanced.shape)

# -------------------- 10. Save processed files --------------------
pd.DataFrame(X_train_balanced, columns=X.columns).to_csv("X_train.csv", index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv("X_test.csv", index=False)
pd.DataFrame(y_train_balanced, columns=[target_col]).to_csv("y_train.csv", index=False)
pd.DataFrame(y_test, columns=[target_col]).to_csv("y_test.csv", index=False)

print("\n💾 Saved: X_train.csv, X_test.csv, y_train.csv, y_test.csv")

# -------------------- 11. Correlation Heatmap --------------------
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
print("📊 Heatmap saved as 'correlation_heatmap.png'")
