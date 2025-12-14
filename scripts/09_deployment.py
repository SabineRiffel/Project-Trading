import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model
model = joblib.load("data/random_forest_model_news.pkl")

# Define FastAPI app
app = FastAPI(title="Stock Return Prediction API")

# Define input schema
class PredictionRequest(BaseModel):
    features: dict  # key-value pairs of feature_name: value

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        X_new = pd.DataFrame([request.features])
        print("Incoming features:", X_new.columns.tolist())
        prediction = model.predict(X_new)[0]
        return {"prediction": float(prediction)}
    except Exception as e:
        return {"error": str(e)}


