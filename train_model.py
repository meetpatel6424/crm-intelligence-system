import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# =========================
# SAMPLE DATASET
# =========================

data = {
    'age': [25, 45, 30, 35, 50, 23, 40, 60],
    'salary': [30000, 80000, 50000, 60000, 90000, 25000, 70000, 100000],
    'experience': [1, 10, 4, 5, 15, 1, 8, 20],
    'pages_visited': [5, 20, 10, 15, 30, 3, 25, 35],
    'emails_opened': [2, 10, 5, 7, 15, 1, 12, 18],
    'previous_purchases': [0, 5, 2, 3, 7, 0, 4, 8],
    'gender': [1, 0, 1, 0, 1, 0, 1, 0],
    'married': [0, 1, 1, 1, 1, 0, 1, 1],
    'lead_source': [0, 2, 1, 0, 2, 3, 1, 2],
    'converted': [0, 1, 0, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# =========================
# FEATURES & TARGET
# =========================

X = df.drop('converted', axis=1)
y = df['converted']

# =========================
# TRAIN MODEL
# =========================

model = RandomForestClassifier()

model.fit(X, y)

# =========================
# SAVE MODEL
# =========================

joblib.dump(model, 'model.pkl')

print("Model saved successfully!")