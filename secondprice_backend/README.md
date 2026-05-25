# SecondPrice — Backend Deployment

Prediksi harga barang bekas via **REST API**, menggunakan model GraphSAGE, GAT, TF-IDF+Ridge, dan XGBoost yang diekstrak dari notebook `SecondPrice_GNN.ipynb`.

---

## Struktur Proyek

```
secondprice_backend/
├── app.py                        # FastAPI server (entry point)
├── train.py                      # Script training & ekspor artefak
├── requirements.txt
│
├── models/
│   └── gnn_models.py             # Arsitektur GraphSAGE & GAT
│
├── utils/
│   ├── preprocessing.py          # Preprocessing & GraphBuilder
│   └── inference.py              # SecondPricePredictor (loader + predict)
│
└── artifacts/                    # Dihasilkan oleh train.py (tidak ada di repo)
    ├── graphsage_secondprice.pt
    ├── gat_secondprice.pt
    ├── tfidf_ridge_pipeline.pkl
    ├── xgboost_model.pkl
    └── graph_meta.pkl
```

---

## Cara Penggunaan

### 1. Install Dependensi

```bash
pip install -r requirements.txt
```

> Untuk GPU (CUDA 12.1):
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> pip install torch-geometric
> ```

### 2. Training & Ekspor Artefak

Letakkan `train2.csv` (atau `train.tsv`) di direktori yang sama, lalu jalankan:

```bash
python train.py --data train2.csv --out artifacts/ --epochs 60
```

Opsi tambahan:
```bash
python train.py \
  --data   train2.csv   \  # path dataset
  --out    artifacts/   \  # direktori output artefak
  --epochs 60           \  # epoch GNN
  --sample 100000          # opsional: batasi N baris (debug)
```

Output di `artifacts/`:
| File | Keterangan |
|---|---|
| `graphsage_secondprice.pt` | Bobot GraphSAGE |
| `gat_secondprice.pt` | Bobot GAT |
| `tfidf_ridge_pipeline.pkl` | TF-IDF + Ridge pipeline |
| `xgboost_model.pkl` | XGBoost model |
| `graph_meta.pkl` | Metadata: encoder, normalisasi, konfigurasi graph |

### 3. Jalankan Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Environment variables opsional:
```bash
ARTIFACTS_DIR=artifacts  # default
DEVICE=auto              # auto | cpu | cuda
```

---

## Dokumentasi API

Setelah server berjalan, buka:
- **Swagger UI** : http://localhost:8000/docs
- **ReDoc**      : http://localhost:8000/redoc

### `POST /predict`

Prediksi harga satu produk.

**Request Body:**
```json
{
  "name":               "Nike Air Max 90 White",
  "brand_name":         "Nike",
  "category_name":      "Men/Shoes/Athletic",
  "item_condition_id":  1,
  "shipping":           0,
  "item_description":   "Size 10. Worn twice. Great condition."
}
```

| Field | Tipe | Default | Keterangan |
|---|---|---|---|
| `name` | string | wajib | Nama produk |
| `brand_name` | string | `""` | Nama merek |
| `category_name` | string | `"Other/Other/Other"` | Format `Main/Sub1/Sub2` |
| `item_condition_id` | int 1–5 | `1` | 1=Baru, 5=Buruk |
| `shipping` | int 0/1 | `0` | 0=Pembeli, 1=Penjual |
| `item_description` | string | `""` | Deskripsi produk |

**Response:**
```json
{
  "graphsage_price":   28.50,
  "gat_price":         27.80,
  "tfidf_ridge_price": 26.40,
  "xgboost_price":     29.10,
  "ensemble_price":    27.80,
  "currency":          "USD"
}
```

> `ensemble_price` adalah **median** dari 4 model (lebih robust terhadap outlier dibanding mean).

---

### `POST /predict/batch`

Prediksi harga banyak produk sekaligus (maks. 100 item).

**Request Body:**
```json
{
  "items": [
    { "name": "Nike Air Max 90", "brand_name": "Nike", ... },
    { "name": "Zara Dress Red",  "brand_name": "Zara", ... }
  ]
}
```

**Response:**
```json
{
  "predictions": [ {...}, {...} ],
  "count": 2
}
```

---

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cpu"
}
```

---

### `GET /models`

Menampilkan informasi semua model yang tersedia.

---

## Contoh `curl`

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Vintage Levi Jeans 501",
    "brand_name": "Levi'\''s",
    "category_name": "Men/Pants/Jeans",
    "item_condition_id": 3,
    "shipping": 1,
    "item_description": "Classic 501. Waist 32, length 30."
  }'
```

---

## Arsitektur Sistem

```
                          ┌─────────────────────────────────────┐
  HTTP Request            │          FastAPI (app.py)           │
  POST /predict  ────────►│                                     │
                          │  PredictRequest (Pydantic schema)   │
                          └────────────────┬────────────────────┘
                                           │
                          ┌────────────────▼────────────────────┐
                          │    SecondPricePredictor             │
                          │       (utils/inference.py)          │
                          │                                     │
                          │  preprocess_single_item()           │
                          │         │                           │
                          │  ┌──────▼──────┐  ┌─────────────┐  │
                          │  │ GraphBuilder│  │  Baseline   │  │
                          │  │ HeteroData  │  │  sklearn/   │  │
                          │  └──────┬──────┘  │  xgboost    │  │
                          │         │         └──────┬──────┘  │
                          │  ┌──────▼──────┐         │         │
                          │  │ GraphSAGE   │         │         │
                          │  │    GAT      │         │         │
                          │  └──────┬──────┘         │         │
                          │         └────────┬────────┘         │
                          │              Ensemble                │
                          └──────────────────┬──────────────────┘
                                             │
                          HTTP Response ◄────┘
                          PredictResponse
```

---

## Catatan Penting

- **Artefak tidak tersedia di repo** — harus di-generate dengan `train.py` menggunakan dataset Mercari.
- Model GNN membutuhkan seluruh graph saat inference (transductive). Untuk produk yang benar-benar baru (belum pernah di-training), baseline TF-IDF+Ridge dan XGBoost memberikan prediksi yang lebih stabil.
- Untuk production, pertimbangkan:
  - **Docker**: containerize dengan `Dockerfile`
  - **HTTPS**: reverse proxy nginx / traefik
  - **Rate limiting**: tambahkan middleware di `app.py`
  - **Monitoring**: Prometheus + Grafana
