from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="SecondPrice API")

class PredictRequest(BaseModel):
    name: str
    brand_name: str = ""
    category_name: str = "Other/Other/Other"
    item_condition_id: int = 1
    shipping: int = 0
    item_description: str = ""

class PredictResponse(BaseModel):
    graphsage_price: float
    gat_price: float
    tfidf_ridge_price: float
    xgboost_price: float
    ensemble_price: float
    currency: str = "USD"

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": False}

@app.get("/models")
def list_models():
    return {"models": [...]}

@app.post("/predict")
def predict(request: PredictRequest):
    # TODO: Implement when models ready
    return {"ensemble_price": 0.0}