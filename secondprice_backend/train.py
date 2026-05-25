"""
SecondPrice — Training Script
Menjalankan ulang pipeline training notebook & mengekspor semua artefak ke disk.

Cara pakai:
    python train.py --data train2.csv --out artifacts/ --epochs 60

Output artefak di --out:
    graphsage_secondprice.pt
    gat_secondprice.pt
    tfidf_ridge_pipeline.pkl
    xgboost_model.pkl
    graph_meta.pkl
"""

import argparse
import os
import pickle
import random
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import xgboost as xgb

from models.gnn_models import GraphSAGERegressor, GATRegressor
from utils.preprocessing import preprocess_dataframe, GraphBuilder

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ── Metrik ────────────────────────────────────────────────────────────────────

def rmsle(y_true, y_pred):
    y_pred = np.maximum(y_pred, 0)
    return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))


def evaluate_model(y_true, y_pred_log, name="Model"):
    y_pred = np.maximum(np.expm1(y_pred_log), 0)
    metrics = {
        "model": name,
        "MAE":   mean_absolute_error(y_true, y_pred),
        "RMSE":  np.sqrt(mean_squared_error(y_true, y_pred)),
        "RMSLE": rmsle(y_true, y_pred),
        "R2":    r2_score(y_true, y_pred),
    }
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:6s}: {v:.4f}")
    return metrics


# ── GNN Training Loop ─────────────────────────────────────────────────────────

def train_gnn(model, data, optimizer, scheduler, n_epochs=60, model_name="GNN"):
    criterion  = nn.MSELoss()
    best_val   = float("inf")
    best_state = None

    for epoch in range(1, n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        loss = criterion(
            out[data["product"].train_mask],
            data["product"].y[data["product"].train_mask],
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out_v = model(data.x_dict, data.edge_index_dict)
            val_loss = criterion(
                out_v[data["product"].val_mask],
                data["product"].y[data["product"].val_mask],
            ).item()

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{model_name}] Epoch {epoch:3d}/{n_epochs} | "
                  f"Train: {loss.item():.4f} | Val: {val_loss:.4f}")

    model.load_state_dict(best_state)
    return model


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🔧 Device   : {DEVICE}")
    print(f"📂 Data     : {args.data}")
    print(f"📦 Output   : {OUT_DIR}")
    print(f"🔁 Epochs   : {args.epochs}")

    # ── 1. Load & Preprocess ─────────────────────────────────────────────────
    print("\n📂 Loading dataset...")
    sep = "\t" if args.data.endswith(".tsv") else ","
    df_raw = pd.read_csv(args.data, sep=sep,
                         nrows=args.sample if args.sample else None,
                         encoding="ISO-8859-1")
    print(f"   Loaded {len(df_raw):,} rows")

    print("⚙️  Preprocessing...")
    t0 = time.time()
    df, le_brand, le_cat_main, le_cat_sub1 = preprocess_dataframe(df_raw)
    print(f"   Done in {time.time()-t0:.1f}s  |  shape: {df.shape}")

    # ── 2. Train / Val / Test Split ───────────────────────────────────────────
    idx = np.arange(len(df))
    idx_train, idx_temp = train_test_split(idx, test_size=0.30, random_state=SEED)
    idx_val,   idx_test = train_test_split(idx_temp, test_size=0.50, random_state=SEED)

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_val   = df.iloc[idx_val].reset_index(drop=True)
    df_test  = df.iloc[idx_test].reset_index(drop=True)
    print(f"   Train:{len(df_train):,}  Val:{len(df_val):,}  Test:{len(df_test):,}")

    results = {}

    # ── 3. Baseline: TF-IDF + Ridge ───────────────────────────────────────────
    print("\n🔵 Training TF-IDF + Ridge...")
    t0 = time.time()
    tfidf_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=50_000, ngram_range=(1, 2),
            min_df=3, sublinear_tf=True,
        )),
        ("ridge", Ridge(alpha=5.0, random_state=SEED)),
    ])
    tfidf_pipeline.fit(df_train["text_features"], df_train["log_price"])
    pred_ridge = tfidf_pipeline.predict(df_test["text_features"])
    print(f"   Done in {time.time()-t0:.1f}s")
    results["TF-IDF + Ridge"] = evaluate_model(
        df_test["price"].values, pred_ridge, "TF-IDF + Ridge"
    )

    # ── 4. Baseline: XGBoost ──────────────────────────────────────────────────
    print("\n🟡 Training XGBoost...")
    XGB_FEAT = ["item_condition_id", "shipping", "brand_id", "cat_main_id", "cat_sub1_id"]
    t0 = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=SEED, n_jobs=-1,
        eval_metric="rmse", early_stopping_rounds=20, verbosity=0,
    )
    xgb_model.fit(
        df_train[XGB_FEAT].values, df_train["log_price"].values,
        eval_set=[(df_val[XGB_FEAT].values, df_val["log_price"].values)],
        verbose=False,
    )
    pred_xgb = xgb_model.predict(df_test[XGB_FEAT].values)
    print(f"   Done in {time.time()-t0:.1f}s")
    results["XGBoost"] = evaluate_model(df_test["price"].values, pred_xgb, "XGBoost")

    # ── 5. Build Heterogeneous Graph ──────────────────────────────────────────
    print("\n🕸️  Building heterogeneous graph...")
    t0 = time.time()

    # TF-IDF vectorizer untuk GNN (128 fitur)
    tfidf_gnn = TfidfVectorizer(max_features=128, ngram_range=(1, 1), sublinear_tf=True)
    tfidf_gnn.fit(df["text_features"])

    # Statistik normalisasi tabular (dari seluruh data)
    TAB_FEATS = ["item_condition_id", "shipping"]
    tab_all   = df[TAB_FEATS].values.astype(np.float32)
    tab_mean  = tab_all.mean(0)
    tab_std   = tab_all.std(0)

    n_brands = int(df["brand_id"].max()) + 1
    n_cats   = int(df["cat_main_id"].max()) + 1

    graph_builder = GraphBuilder(tfidf_gnn, n_brands, n_cats)
    graph_data    = graph_builder.build_from_df(df, device=DEVICE)

    # Masks
    N = len(df)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)
    train_mask[idx_train] = True
    val_mask[idx_val]     = True
    test_mask[idx_test]   = True
    graph_data["product"].train_mask = train_mask.to(DEVICE)
    graph_data["product"].val_mask   = val_mask.to(DEVICE)
    graph_data["product"].test_mask  = test_mask.to(DEVICE)

    in_channels = {
        "product":  graph_data["product"].x.shape[1],
        "brand":    graph_data["brand"].x.shape[1],
        "category": graph_data["category"].x.shape[1],
    }
    print(f"   Done in {time.time()-t0:.1f}s")
    print(f"   in_channels: {in_channels}")

    # ── 6. GraphSAGE ──────────────────────────────────────────────────────────
    print("\n🔵 Training GraphSAGE...")
    t0 = time.time()
    sage = GraphSAGERegressor(
        in_channels_dict=in_channels, hidden_channels=128, dropout=0.3,
        edge_types=list(graph_data.edge_types),
    ).to(DEVICE)
    sage, _ = _fit_gnn(sage, graph_data, args.epochs, "GraphSAGE")
    print(f"   Done in {time.time()-t0:.1f}s")

    sage.eval()
    with torch.no_grad():
        pred_sage = sage(graph_data.x_dict, graph_data.edge_index_dict)
        pred_sage = pred_sage[graph_data["product"].test_mask].cpu().numpy()
    results["GraphSAGE"] = evaluate_model(df_test["price"].values, pred_sage, "GraphSAGE")

    # ── 7. GAT ────────────────────────────────────────────────────────────────
    print("\n🟢 Training GAT...")
    t0 = time.time()
    gat = GATRegressor(
        in_channels_dict=in_channels, hidden_channels=64, heads=4, dropout=0.3,
        edge_types=list(graph_data.edge_types),
    ).to(DEVICE)
    gat, _ = _fit_gnn(gat, graph_data, args.epochs, "GAT", lr=3e-3)
    print(f"   Done in {time.time()-t0:.1f}s")

    gat.eval()
    with torch.no_grad():
        pred_gat = gat(graph_data.x_dict, graph_data.edge_index_dict)
        pred_gat = pred_gat[graph_data["product"].test_mask].cpu().numpy()
    results["GAT"] = evaluate_model(df_test["price"].values, pred_gat, "GAT")

    # ── 8. Simpan Artefak ─────────────────────────────────────────────────────
    print("\n💾 Menyimpan artefak...")

    torch.save(sage.state_dict(), OUT_DIR / "graphsage_secondprice.pt")
    torch.save(gat.state_dict(),  OUT_DIR / "gat_secondprice.pt")

    with open(OUT_DIR / "tfidf_ridge_pipeline.pkl", "wb") as f:
        pickle.dump(tfidf_pipeline, f)
    with open(OUT_DIR / "xgboost_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)

    graph_meta = {
        "n_brands":    n_brands,
        "n_cats":      n_cats,
        "in_channels": in_channels,
        "tab_mean":    tab_mean,
        "tab_std":     tab_std,
        "le_brand":    le_brand,
        "le_cat_main": le_cat_main,
        "le_cat_sub1": le_cat_sub1,
        "tfidf_vec":   tfidf_gnn,
    }
    with open(OUT_DIR / "graph_meta.pkl", "wb") as f:
        pickle.dump(graph_meta, f)

    print(f"\n✅ Semua artefak disimpan di '{OUT_DIR}':")
    for p in sorted(OUT_DIR.iterdir()):
        size_kb = p.stat().st_size / 1024
        print(f"   {p.name:45s} {size_kb:8.1f} KB")

    print("\n🏆 Ringkasan Performa:")
    results_df = pd.DataFrame(list(results.values())).set_index("model")
    print(results_df.sort_values("RMSLE").to_string(float_format=lambda x: f"{x:.4f}"))


def _fit_gnn(model, data, n_epochs, name, lr=5e-3):
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    model = train_gnn(model, data, optimizer, scheduler, n_epochs=n_epochs, model_name=name)
    return model, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SecondPrice — Training Script")
    parser.add_argument("--data",    type=str, default="train2.csv", help="Path ke file dataset (.csv / .tsv)")
    parser.add_argument("--out",     type=str, default="artifacts",  help="Direktori output artefak")
    parser.add_argument("--epochs",  type=int, default=60,           help="Jumlah epoch GNN")
    parser.add_argument("--sample",  type=int, default=None,         help="Batasi N baris (opsional, untuk debugging)")
    args = parser.parse_args()
    main(args)
