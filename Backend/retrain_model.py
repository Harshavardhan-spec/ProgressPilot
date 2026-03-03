import pandas as pd
import joblib
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier


# Mongo connection

MONGO_URL = "mongodb+srv://Harsha_Vinay:VH12@cluster0.vw91y2y.mongodb.net/?appName=Cluster0"

client = MongoClient(MONGO_URL)
db = client["ai_learning"]
collection = db["recommendations"]

print("Fetching training data from Mongodb...")

data = list(collection.find({}, {"_id": 0}))

if len(data) < 20:
    print("Not enough data to retrain")
    exit()

df = pd.DataFrame(data)


# Build training dataset


X = pd.json_normalize(df["student_input"])
y = df["recommended_course"]

# one-hot encode categorical columns
X = pd.get_dummies(X)


# Train model


print("🧠 Retraining model...")

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)


# Save model


joblib.dump(model, "model.pkl") # this is model.pkl file
joblib.dump(X.columns.tolist(), "columns.pkl") # this is model.column pkl file

print("Model retrained and saved!") #if retrained...