import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Setup logger
logger = logging.getLogger(__name__)

app = FastAPI(title="SecondPrice API")

# ===== Models (Request & Response) =====
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

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

class BatchPredictRequest(BaseModel):
    items: list[PredictRequest]

class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    count: int

# ===== Initialize Predictor =====
predictor = None

# ===== Routes =====
@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Cek status server dan apakah model sudah dimuat."""
    return {
        "status":       "ok" if predictor else "model_not_loaded",
        "model_loaded": predictor is not None,
        "device":       str(predictor.device) if predictor else "N/A",
    }

@app.get("/models")
def list_models():
    return {"models": [...]}

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    """
    Prediksi harga untuk **satu produk**.

    Mengembalikan prediksi dari semua model (GraphSAGE, GAT, TF-IDF+Ridge, XGBoost)
    beserta nilai ensemble (rata-rata) sebagai rekomendasi harga jual.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat, coba lagi sebentar.")

    try:
        result = predictor.predict(
            name=request.name,
            brand_name=request.brand_name or "",
            category_name=request.category_name or "Other/Other/Other",
            item_condition_id=request.item_condition_id,
            shipping=request.shipping,
            item_description=request.item_description or "",
        )
        return {**result, "currency": "USD"}

    except Exception as exc:
        logger.error(f"Prediction error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(exc)}")


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch(request: BatchPredictRequest):
    """
    Prediksi harga untuk **banyak produk sekaligus** (maks. 100 item).
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model belum dimuat, coba lagi sebentar.")

    try:
        predictions = []
        for item in request.items:
            result = predictor.predict(
                name=item.name,
                brand_name=item.brand_name or "",
                category_name=item.category_name or "Other/Other/Other",
                item_condition_id=item.item_condition_id,
                shipping=item.shipping,
                item_description=item.item_description or "",
            )
            predictions.append({**result, "currency": "USD"})

        return {"predictions": predictions, "count": len(predictions)}

    except Exception as exc:
        logger.error(f"Batch prediction error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(exc)}")
