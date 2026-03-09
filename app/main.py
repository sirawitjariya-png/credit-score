from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.scoring import score_customer

app = FastAPI(
    title="Credit Scoring API",
    description="API for predicting credit score using trained model",
    version="1.0"
)


class CustomerData(BaseModel):
    data: List[dict]


@app.get("/")
def home():
    return {"message": "Credit Scoring API is running"}


@app.post("/score")
def score(payload: CustomerData):

    print("INPUT RECEIVED:")
    print(payload.data)

    predictions = score_customer(payload.data)

    return {
        "scores": predictions
    }