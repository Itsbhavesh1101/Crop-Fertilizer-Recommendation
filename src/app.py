from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(
    title="Crop & Fertilizer Recommendation API",
    description="API for recommending crops and fertilizers based on soil and environmental parameters.",
    version="1.0.0"
)

# Load Models and Scalers
try:
    with open(os.path.join("models", "crop_model.sav"), "rb") as f:
        crop_model = pickle.load(f)
    with open(os.path.join("models", "crop_scaler.sav"), "rb") as f:
        crop_scaler = pickle.load(f)
    with open(os.path.join("models", "crop_dict.pkl"), "rb") as f:
        crop_dict = pickle.load(f)

    with open(os.path.join("models", "fertilizer_model.sav"), "rb") as f:
        fert_model = pickle.load(f)
    with open(os.path.join("models", "fertilizer_scaler.sav"), "rb") as f:
        fert_scaler = pickle.load(f)
    with open(os.path.join("models", "fertilizer_artifacts.pkl"), "rb") as f:
        fert_artifacts = pickle.load(f)
except Exception as e:
    print(f"Error loading models. Did you run train.py? Error: {e}")

class CropRequest(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

class FertilizerRequest(BaseModel):
    temperature: float
    humidity: float
    moisture: float
    soil_type: str
    crop_type: str
    N: float
    K: float
    P: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop & Fertilizer Recommendation API. Visit /docs for API documentation."}

@app.post("/predict_crop")
def predict_crop(request: CropRequest):
    try:
        features = np.array([[
            request.N, request.P, request.K, 
            request.temperature, request.humidity, 
            request.ph, request.rainfall
        ]])
        features_scaled = crop_scaler.transform(features)
        prediction = crop_model.predict(features_scaled)
        recommended_crop = crop_dict.get(prediction[0], "Unknown")
        return {"recommended_crop": recommended_crop}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_fertilizer")
def predict_fertilizer(request: FertilizerRequest):
    try:
        le_soil = fert_artifacts["le_soil"]
        le_crop = fert_artifacts["le_crop"]
        inv_fert_dict = fert_artifacts["inv_fert_dict"]

        try:
            soil_encoded = le_soil.transform([request.soil_type])[0]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid soil type. Valid types are: {list(le_soil.classes_)}")
        
        try:
            crop_encoded = le_crop.transform([request.crop_type])[0]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid crop type. Valid types are: {list(le_crop.classes_)}")

        features = np.array([[
            request.temperature, request.humidity, request.moisture,
            soil_encoded, crop_encoded, 
            request.N, request.K, request.P
        ]])
        
        features_scaled = fert_scaler.transform(features)
        prediction = fert_model.predict(features_scaled)
        recommended_fertilizer = inv_fert_dict.get(prediction[0], "Unknown")
        
        return {"recommended_fertilizer": recommended_fertilizer}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
