from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pymongo import MongoClient
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
import subprocess
#This code is for integrating ml and 
def trigger_retraining():
    try:
        print("🚀 Triggering model retraining...")
        subprocess.Popen(["python", "retrain_model.py"])
    except Exception as e:
        print("Retraining failed:", e)

COURSE_RESOURCES = {
    "Math Foundation": {
        "videos": [
            {"title": "Algebra Basics", "url": "https://youtube.com/playlist?list=PLU_DCVXL8MyOSL8b-E0NVkhyjk9FLWYJW&si=jgw0sbvjjGTyAXgW"},
            {"title": "Quadratic Equations", "url": "https://www.youtube.com/playlist?list=PLU_DCVXL8MyMZ5nO2RajVuRWGRBt5XxQb"}
        ],
        "books": [
            {"title": "NCERT Mathematics Class"},
            {"title": "RD Sharma Algebra"}
        ]
    },
    "JEE Physics": {
        "videos": [
            {"title": "Kinematics Full Course", "url": "https://www.youtube.com/watch?v=b1t41Q3xRM8&list=PL0o_zxa4K1BU6wPPLDsoTj1_wEf0LSNeR"}
        ],
        "books": [
            {"title": "HC Verma Vol 1"}
        ]
    },
    "Chemistry booster": {
        "videos": [
            {"title": "Chemical Reactions Full Course", "url": "https://www.youtube.com/watch?v=EhVf0hkY4xc&list=PLVLoWQFkZbhV8gdBQOVjUpLW_3jwak-uA"}
        ],
        "books": [
            {"title": "RCTV Vol 1"}
        ]
    }
}

app = FastAPI(title="Project_M")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# =============================
# MongoDB connection for our proj
# =============================
MONGO_URL = "mongodb+srv://Harsha_Vinay:VH12@cluster0.vw91y2y.mongodb.net/?appName=Cluster0"

#the mongo block 
client = MongoClient(MONGO_URL)
db = client["ai_learning"]
collection = db["recommendations"]
feedback_collection = db["feedback"]

#Loading  ml - model
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")
class StudentInput(BaseModel):
    class_level: int
    weak_subject: str
    learning_style: str
    goal: str
class FeedbackInput(BaseModel):
    recommendation_id: str
    rating: str   # "helpful" or "not_helpful"
    comment: str | None = None

@app.get("/")
def home():
    return {"message": "Backend is running successfully"}

@app.post("/recommend")
def recommend(student: StudentInput):
    # 1️ Build input
    input_dict = {
        "class_level": student.class_level,
        f"weak_subject_{student.weak_subject}": 1,
        f"learning_style_{student.learning_style}": 1
    }

    # 2️ Create dataframe
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=columns, fill_value=0)

    # 3️ Prediction
    prediction = model.predict(input_df)[0]

    # Vid LINE
    resources = COURSE_RESOURCES.get(prediction, {})

    # 4️ Confidence
    proba = model.predict_proba(input_df)[0]
    confidence = max(proba)

    # 5️ Save to Mongo
    result = collection.insert_one({
        "student_input": student.dict(),
        "recommended_course": prediction,
        "confidence": float(confidence),
        "timestamp": datetime.utcnow()
    })

    # 6️⃣ Return response
    return {
        "recommendation_id": str(result.inserted_id),
        "recommended_course": prediction,
        "priority": "High",
        "confidence": round(float(confidence), 3),
        "resources": resources
    }

@app.post("/feedback")
def submit_feedback(feedback: FeedbackInput):
    feedback_collection.insert_one({
        "recommendation_id": feedback.recommendation_id,
        "rating": feedback.rating,
        "comment": feedback.comment,
        "timestamp": datetime.utcnow()
    })

    # checking feedback count (since ml should get retrained for every 20 feedbacks..)
    feedback_count = feedback_collection.count_documents({})

    if feedback_count > 0 and feedback_count % 20 == 0:
        trigger_retraining()

    return {"message": "Feedback recorded successfully"}


#Analytucs 

@app.get("/analytics")
def get_analytics():
    total_recs = collection.count_documents({})
    total_feedback = feedback_collection.count_documents({})

    helpful = feedback_collection.count_documents({"rating": "helpful"})
    not_helpful = feedback_collection.count_documents({"rating": "not_helpful"})

    helpful_pct = (helpful / total_feedback * 100) if total_feedback else 0
    not_helpful_pct = (not_helpful / total_feedback * 100) if total_feedback else 0

    # Most recommended course
    pipeline = [
        {"$group": {"_id": "$recommended_course", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 1}
    ]
    top_course = list(collection.aggregate(pipeline))
    top_course_name = top_course[0]["_id"] if top_course else "N/A"

    # Average confidence
    avg_conf_pipeline = [
        {"$group": {"_id": None, "avg_conf": {"$avg": "$confidence"}}}
    ]
    avg_conf = list(collection.aggregate(avg_conf_pipeline))
    avg_conf_val = avg_conf[0]["avg_conf"] if avg_conf else 0

    # Course distribution for bar chart
    course_pipeline = [
    {"$group": {"_id": "$recommended_course", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
    ]

    courses = list(collection.aggregate(course_pipeline))

    return {
    "total_recommendations": total_recs,
    "total_feedback": total_feedback,
    "helpful_percentage": round(helpful_pct, 2),
    "not_helpful_percentage": round(not_helpful_pct, 2),
    "top_course": top_course_name,
    "average_confidence": round(avg_conf_val, 3),
    "courses": courses
}