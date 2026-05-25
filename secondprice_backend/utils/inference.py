"""
SecondPrice — Model Loader & Inference Engine

Bertanggung-jawab untuk:
  1. Memuat semua artefak model dari disk (GNN .pt + baseline .pkl)
  2. Menyediakan fungsi predict() yang menerima satu item produk
     dan mengembalikan prediksi harga dari seluruh model

Cara pakai:
    from utils.inference import SecondPricePredictor
    predictor = SecondPricePredictor("artifacts/")
    result    = predictor.predict(name="...", brand_name="...", ...)
"""

import os
import pickle
import numpy as np
import torch
from pathlib import Path

from models.gnn_models import GraphSAGERegressor, GATRegressor, EDGE_TYPES
from utils.preprocessing import preprocess_single_item, GraphBuilder


# ── Konstanta (harus sinkron dengan training) ──────────────────────────────────
GRAPHSAGE_HIDDEN = 128
GAT_HIDDEN       = 64
GAT_HEADS        = 4
TFIDF_MAX_FEAT   = 128
DROPOUT          = 0.0   # matikan dropout saat inference


class SecondPricePredictor:
    """
    High-level predictor yang menggabungkan semua model SecondPrice.

    Struktur direktori artefak yang diharapkan
    ------------------------------------------
    artifacts/
    ├── graphsage_secondprice.pt      # state dict GraphSAGE
    ├── gat_secondprice.pt            # state dict GAT
    ├── tfidf_ridge_pipeline.pkl      # sklearn Pipeline (TfidfVectorizer + Ridge)
    ├── xgboost_model.pkl             # XGBRegressor
    └── graph_meta.pkl                # metadata graph (lihat keterangan di bawah)

    graph_meta.pkl berisi dict dengan kunci:
        n_brands      : int
        n_cats        : int
        in_channels   : dict {"product": int, "brand": int, "category": int}
        tab_mean      : np.ndarray  shape (2,)
        tab_std       : np.ndarray  shape (2,)
        le_brand      : LabelEncoder
        le_cat_main   : LabelEncoder
        le_cat_sub1   : LabelEncoder
        tfidf_vec     : TfidfVectorizer (128 fitur, fit pada training set)
    """

    def __init__(self, artifacts_dir: str = "artifacts", device: str = "auto"):
        self.artifacts_dir = Path(artifacts_dir)
        self.device = torch.device(
            "cuda" if (device == "auto" and torch.cuda.is_available())
            else ("cpu" if device == "auto" else device)
        )
        self._load_artifacts()
        print(f"✅ SecondPricePredictor siap | device: {self.device}")

    # ── Loader ────────────────────────────────────────────────────────────────

    def _load_artifacts(self):
        """Muat semua artefak dari disk."""
        meta_path = self.artifacts_dir / "graph_meta.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"graph_meta.pkl tidak ditemukan di '{self.artifacts_dir}'. "
                "Jalankan train.py terlebih dahulu untuk menghasilkan artefak."
            )

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        self.le_brand    = meta["le_brand"]
        self.le_cat_main = meta["le_cat_main"]
        self.le_cat_sub1 = meta["le_cat_sub1"]
        self.tab_mean    = meta["tab_mean"]
        self.tab_std     = meta["tab_std"]
        n_brands         = meta["n_brands"]
        n_cats           = meta["n_cats"]
        in_channels      = meta["in_channels"]
        tfidf_vec        = meta["tfidf_vec"]

        # GraphBuilder (untuk inference graph)
        self.graph_builder = GraphBuilder(tfidf_vec, n_brands, n_cats)

        # ── GNN Models ────────────────────────────────────────────────────────
        self.graphsage = GraphSAGERegressor(
            in_channels_dict=in_channels,
            hidden_channels=GRAPHSAGE_HIDDEN,
            dropout=DROPOUT,
            edge_types=EDGE_TYPES,
        ).to(self.device)

        self.gat = GATRegressor(
            in_channels_dict=in_channels,
            hidden_channels=GAT_HIDDEN,
            heads=GAT_HEADS,
            dropout=DROPOUT,
            edge_types=EDGE_TYPES,
        ).to(self.device)

        self._load_state_dict(self.graphsage, "graphsage_secondprice.pt")
        self._load_state_dict(self.gat,       "gat_secondprice.pt")
        self.graphsage.eval()
        self.gat.eval()

        # ── Baseline Models ───────────────────────────────────────────────────
        self.tfidf_ridge = self._load_pickle("tfidf_ridge_pipeline.pkl")
        self.xgb_model   = self._load_pickle("xgboost_model.pkl")

    def _load_state_dict(self, model: torch.nn.Module, filename: str):
        path = self.artifacts_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Bobot model '{filename}' tidak ditemukan di '{self.artifacts_dir}'.")
        state = torch.load(path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)

    def _load_pickle(self, filename: str):
        path = self.artifacts_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"File pickle '{filename}' tidak ditemukan di '{self.artifacts_dir}'.")
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        name: str,
        brand_name: str = "",
        category_name: str = "Other/Other/Other",
        item_condition_id: int = 1,
        shipping: int = 0,
        item_description: str = "",
    ) -> dict:
        """
        Prediksi harga untuk satu produk baru.

        Parameters
        ----------
        name              : nama produk
        brand_name        : nama merek (kosong → 'Unknown')
        category_name     : format 'Main/Sub1/Sub2'
        item_condition_id : 1 (baru) – 5 (buruk)
        shipping          : 0 (buyer) / 1 (seller)
        item_description  : deskripsi teks

        Returns
        -------
        dict dengan kunci:
            graphsage_price, gat_price,
            tfidf_ridge_price, xgboost_price,
            ensemble_price  (rata-rata keempat model)
        """
        # 1. Preprocessing
        item = preprocess_single_item(
            name=name,
            brand_name=brand_name,
            category_name=category_name,
            item_condition_id=item_condition_id,
            shipping=shipping,
            item_description=item_description,
            le_brand=self.le_brand,
            le_cat_main=self.le_cat_main,
            le_cat_sub1=self.le_cat_sub1,
        )

        # 2. GNN predictions
        graph_data = self.graph_builder.build_inference(
            item, self.tab_mean, self.tab_std, device=self.device
        )

        with torch.no_grad():
            sage_log = self.graphsage(
                graph_data.x_dict, graph_data.edge_index_dict
            )[0].item()

            gat_log = self.gat(
                graph_data.x_dict, graph_data.edge_index_dict
            )[0].item()

        sage_price = float(np.expm1(max(sage_log, 0)))
        gat_price  = float(np.expm1(max(gat_log,  0)))

        # 3. Baseline predictions
        ridge_log = float(self.tfidf_ridge.predict([item["text_features"]])[0])
        ridge_price = float(np.expm1(max(ridge_log, 0)))

        xgb_features = [[
            item["item_condition_id"],
            item["shipping"],
            item["brand_id"],
            item["cat_main_id"],
            item["cat_sub1_id"],
        ]]
        xgb_log   = float(self.xgb_model.predict(xgb_features)[0])
        xgb_price = float(np.expm1(max(xgb_log, 0)))

        # 4. Ensemble (median of 4 models)
        prices = [sage_price, gat_price, ridge_price, xgb_price]
        ensemble_price = round(float(np.median(prices)), 2)

        return {
            "graphsage_price":   round(sage_price,  2),
            "gat_price":         round(gat_price,   2),
            "tfidf_ridge_price": round(ridge_price, 2),
            "xgboost_price":     round(xgb_price,   2),
            "ensemble_price":    ensemble_price,
        }

    def predict_batch(self, items: list[dict]) -> list[dict]:
        """Convenience wrapper untuk prediksi banyak item sekaligus."""
        return [self.predict(**item) for item in items]
