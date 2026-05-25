"""
SecondPrice — FastAPI Backend Server

Endpoint:
  POST /predict          — prediksi harga satu produk
  POST /predict/batch    — prediksi harga banyak produk sekaligus
  GET  /health           — cek status server & model
  GET  /models           — info model yang dimuat

Cara menjalankan:
  uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Environment variables:
  ARTIFACTS_DIR   : path ke direktori artefak (default: artifacts)
  DEVICE          : cpu | cuda | auto (default: auto)
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from utils.inference import SecondPricePredictor

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Global predictor (dimuat sekali saat startup) ─────────────────────────────
predictor: SecondPricePredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts saat startup, cleanup saat shutdown."""
    global predictor
    artifacts_dir = os.getenv("ARTIFACTS_DIR", "artifacts")
    device        = os.getenv("DEVICE", "auto")

    logger.info(f"Loading model artifacts dari '{artifacts_dir}'...")
    t0 = time.time()
    predictor = SecondPricePredictor(artifacts_dir=artifacts_dir, device=device)
    logger.info(f"Model siap dalam {time.time()-t0:.2f}s")

    yield  # aplikasi berjalan

    logger.info("Server shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SecondPrice API",
    description=(
        "Prediksi harga barang bekas menggunakan Graph Neural Network (GraphSAGE & GAT) "
        "dibandingkan dengan baseline TF-IDF+Ridge dan XGBoost."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=500, example="Nike Air Max 90 White")
    brand_name: Optional[str] = Field(default="", example="Nike")
    category_name: Optional[str] = Field(
        default="Other/Other/Other",
        example="Men/Shoes/Athletic",
    )
    item_condition_id: int = Field(
        default=1, ge=1, le=5,
        description="Kondisi barang: 1=Baru, 2=Baru tanpa tag, 3=Baik, 4=Cukup, 5=Buruk",
    )
    shipping: int = Field(
        default=0, ge=0, le=1,
        description="Siapa yang membayar ongkir: 0=Pembeli, 1=Penjual",
    )
    item_description: Optional[str] = Field(
        default="",
        max_length=5000,
        example="Size 10. Worn twice. Great condition.",
    )

    @field_validator("item_condition_id")
    @classmethod
    def validate_condition(cls, v):
        if v not in range(1, 6):
            raise ValueError("item_condition_id harus antara 1–5")
        return v


class PredictResponse(BaseModel):
    graphsage_price:   float
    gat_price:         float
    tfidf_ridge_price: float
    xgboost_price:     float
    ensemble_price:    float
    currency:          str = "USD"

    model_config = {"json_schema_extra": {
        "example": {
            "graphsage_price":   28.50,
            "gat_price":         27.80,
            "tfidf_ridge_price": 26.40,
            "xgboost_price":     29.10,
            "ensemble_price":    27.95,
            "currency":          "USD",
        }
    }}


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Cek status server dan apakah model sudah dimuat."""
    return {
        "status":       "ok" if predictor else "model_not_loaded",
        "model_loaded": predictor is not None,
        "device":       str(predictor.device) if predictor else "N/A",
    }


@app.get("/models", tags=["System"])
def list_models():
    """Informasi model yang tersedia."""
    return {
        "models": [
            {
                "id":          "graphsage",
                "name":        "GraphSAGE",
                "type":        "GNN",
                "description": "Heterogeneous GraphSAGE — inductive, cocok untuk produk baru.",
            },
            {
                "id":          "gat",
                "name":        "GAT",
                "type":        "GNN",
                "description": "Graph Attention Network — memberi bobot lebih pada tetangga relevan.",
            },
            {
                "id":          "tfidf_ridge",
                "name":        "TF-IDF + Ridge",
                "type":        "Baseline",
                "description": "NLP klasik — cepat dan mudah diinterpretasi.",
            },
            {
                "id":          "xgboost",
                "name":        "XGBoost",
                "type":        "Baseline",
                "description": "Gradient boosting berbasis fitur tabular.",
            },
            {
                "id":          "ensemble",
                "name":        "Ensemble",
                "type":        "Ensemble",
                "description": "Rata-rata dari keempat model di atas.",
            },
        ]
    }


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
