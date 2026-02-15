from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

model = joblib.load("models/model.pkl")

class HeartInput(BaseModel):
    Age: float
    Sex: float
    Chest_pain_type: float
    BP: float
    Cholesterol: float
    FBS_over_120: float
    EKG_results: float
    Max_HR: float
    Exercise_angina: float
    ST_depression: float
    Slope_of_ST: float
    Number_of_vessels_fluro: float
    Thallium: float

@app.post("/predict")
def predict(data: HeartInput):
    df = pd.DataFrame([data.dict()])
    prediction = model.predict(df)[0]
    return {"prediction": prediction}
