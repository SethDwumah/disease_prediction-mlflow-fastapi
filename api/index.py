# model_serve.py
from fastapi import FastAPI
import os
from pydantic import BaseModel
from typing import List
from datetime import datetime
import joblib
import mlflow
from feature_processing import *
# Define input schema

class InputData(BaseModel):
    text: str

app = FastAPI()

# Load model from MLflow Registry
MODEL_URI = "models:/Symptom-Disease-MNB/5"   # <-- no /api/ prefix
model = mlflow.sklearn.load_model(MODEL_URI)

# Load artifacts from bundle
BASE_DIR = Path(__file__).resolve().parent
vect = joblib.load(BASE_DIR / "Vectorizer.pkl")
lab_encoder = joblib.load(BASE_DIR / "label_encoder.pkl")

@app.get("/")
def home():
    return {"message": "Symptom-Disease API is running. Use POST /predict."}

@app.post("/predict")
def predict(input: InputData):
    text = input.text
    text =clean_text_for_prediction(text)
    emb_text = vect.transform([text])
    prediction = model.predict(emb_text)
    label = lab_encoder.inverse_transform(prediction)
 
    return {"predictions":label[0] }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
