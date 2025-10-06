# model_serve.py
from fastapi import FastAPI
import mlflow
import pandas as pd
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

model_url = "models:/Symptom-Disease-MNB/5" #version 5
load_model = mlflow.sklearn.load_model(model_url)

vect = joblib.load('Vectorizer.pkl','rb')
lab_encoder = joblib.load('label_encoder.pkl','rb')


@app.get("/")
def home():
    return {"message": "Symptom-Disease API is running. Use POST /predict."}

@app.post("/predict")
def predict(input: InputData):
    text = input.text
    text =clean_text_for_prediction(text)
    emb_text = vect.transform([text])
    prediction = load_model.predict(emb_text)
    label = lab_encoder.inverse_transform(prediction)
 
    return {"predictions":label[0] }
