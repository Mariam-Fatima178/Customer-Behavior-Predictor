
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np

# Load saved model
model = load("model.joblib")

# Create FastAPI app
app = FastAPI()

# Define input structure
class InputData(BaseModel):
    age: int
    income: int

@app.post("/predict")
def predict(data: InputData):
    X = np.array([[data.age, data.income]])
    prediction = model.predict(X)
    return {"will_buy": bool(prediction[0])}
