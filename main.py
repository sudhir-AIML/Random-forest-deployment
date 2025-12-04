from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Iris Model API")

# Load model
model = joblib.load("iris_model.joblib")


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Model API"}


@app.post("/predict")
def predict(input_data: IrisInput):
    data = np.array(
        [
            [
                input_data.sepal_length,
                input_data.sepal_width,
                input_data.petal_length,
                input_data.petal_width,
            ]
        ]
    )
    prediction = model.predict(data)
    return {"prediction": int(prediction[0])}
