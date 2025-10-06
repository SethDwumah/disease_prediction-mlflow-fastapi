import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from feature_processing import clean_text
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import mlflow
import mlflow.sklearn

alpha = 1e-1
fit_prior = False

# Load data
df = pd.read_csv("Symptom2Disease.csv")
if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Preprocess
preprocessed_df = clean_text(df, "text")

label_encoder = LabelEncoder()
preprocessed_df["label_encoded"] = label_encoder.fit_transform(preprocessed_df["label"])

train_df, test_df = train_test_split(
    preprocessed_df, test_size=0.3, stratify=preprocessed_df["label"], shuffle=True, random_state=101
)

x_train = train_df["text"].tolist()
x_test  = test_df["text"].tolist()
y_train = train_df["label_encoded"].values
y_test  = test_df["label_encoded"].values

vector = TfidfVectorizer()
x_train_emb = vector.fit_transform(x_train)
x_test_emb  = vector.transform(x_test)

def eval_model(y_true, y_pred):
    accuracy  = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, average="macro", zero_division=0)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="macro", zero_division=0)
    return accuracy, f1, precision, recall

# Create experiment & run
sym_xp = mlflow.set_experiment("Disease symptom experiment")
with mlflow.start_run(experiment_id=sym_xp.experiment_id):
    # Train model
    mn = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
    mn.fit(x_train_emb, y_train)

    # Predict & evaluate
    y_pred = mn.predict(x_test_emb)
    accuracy, f1, precision, recall = eval_model(y_test, y_pred)

    # Persist preprocessing assets locally (so we can log them as artifacts)
    vectorizer_path = "Vectorizer.pkl"
    label_en_path = "label_encoder.pkl"
    with open(vectorizer_path, "wb") as f:
        joblib.dump(vector, f)
    with open(label_en_path, "wb") as f:
        joblib.dump(label_encoder, f)

    # Log params/metrics/artifacts **inside** the active run
    mlflow.log_params({
        "Alpha": alpha,
        "Fit_prior": fit_prior,
        "Vectorizer": "TfidfVectorizer"
    })
    mlflow.log_metrics({
        "Accuracy": accuracy,
        "F1": f1,
        "Precision": precision,
        "Recall": recall
    })
    mlflow.log_artifact(vectorizer_path)
    mlflow.log_artifact(label_en_path)

    # Log model (fix: pass artifact_path only once)
    mlflow.sklearn.log_model(
        sk_model=mn,
        artifact_path="model",
        registered_model_name="Symptom-Disease-MNB"   # works if registry is enabled
    )

print(f"MultinomialNB: alpha={alpha}, fit_prior={fit_prior}")
print(f"Accuracy: {accuracy:.3f} | F1: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")
