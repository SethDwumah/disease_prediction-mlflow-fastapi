import joblib
import mlflow
import pandas as pd
from feature_processing import *
model_url = "models:/Symptom-Disease-MNB/5" #version 5
load_model = mlflow.sklearn.load_model(model_url)

vect = joblib.load('Vectorizer.pkl','rb'))
lab_encoder = joblib.load('label_encoder.pkl','rb')

text = "Running stomach and headache is killing me"
text =clean_text_for_prediction(text)
emb_text = vect.transform([text])
prediction = load_model.predict(emb_text)
label = lab_encoder.inverse_transform(prediction)
print(label[0])