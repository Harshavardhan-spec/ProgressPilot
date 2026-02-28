import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ==============================
# 1. Create larger synthetic dataset
# ==============================
np.random.seed(42)

classes = [8, 9, 10, 11, 12]
subjects = ["Math", "Science", "Physics", "Chemistry"]
styles = ["video", "text"]

data_rows = []

for _ in range(500): 
    cls = np.random.choice(classes)
    subj = np.random.choice(subjects)
    style = np.random.choice(styles)

    # simple rule logic (can improve later)
    if subj == "Math":
        course = "Math Foundation"
    elif subj == "Physics":
        course = "JEE Physics"
    elif subj == "Chemistry":
        course = "Chemistry Booster"
    else:
        course = "Science Basics"

    data_rows.append([cls, subj, style, course])

df = pd.DataFrame(
    data_rows,
    columns=["class_level", "weak_subject", "learning_style", "recommended_course"]
)
print("\n📊 Class distribution:")
print(df["recommended_course"].value_counts())
# ==============================
# 2. Encode categorical features
# ==============================
df_encoded = pd.get_dummies(df, columns=["weak_subject", "learning_style"])

X = df_encoded.drop("recommended_course", axis=1)
y = df_encoded["recommended_course"]

# ==============================
# 3. Train/test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 4. Train Random Forest
# ==============================
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# ==============================
# 5. Evaluate model
# ==============================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# 6. Save model
# ==============================
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "columns.pkl")

print("\n✅ Improved model saved!")