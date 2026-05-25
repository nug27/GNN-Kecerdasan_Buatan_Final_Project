"""
SecondPrice — Preprocessing & Feature Engineering
Diekstrak dari SecondPrice_GNN.ipynb

Berisi:
  - preprocess_single   : preprocessing satu input produk (inference)
  - build_product_vector: membuat feature vector produk (TF-IDF + tabular)
  - GraphBuilder        : membangun HeteroData graph dari DataFrame
"""

import re
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from sklearn.preprocessing import LabelEncoder


# ── Konstanta ──────────────────────────────────────────────────────────────────
TFIDF_MAX_FEATURES = 128  # harus sama dengan saat training
TAB_FEATURES       = ["item_condition_id", "shipping"]


def clean_text(text: str) -> str:
    """Membersihkan teks: lowercase, hapus karakter non-alfanumerik, strip."""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def decompose_category(category_name: str) -> tuple[str, str, str]:
    """
    Memecah 'Main/Sub1/Sub2' menjadi tiga level kategori.
    Mengembalikan ('Other', 'Other', 'Other') jika format tidak sesuai.
    """
    parts = str(category_name).split("/", 2)
    main  = parts[0].strip() if len(parts) > 0 else "Other"
    sub1  = parts[1].strip() if len(parts) > 1 else "Other"
    sub2  = parts[2].strip() if len(parts) > 2 else "Other"
    return main or "Other", sub1 or "Other", sub2 or "Other"


def preprocess_dataframe(df: pd.DataFrame):
    """
    Pipeline preprocessing lengkap untuk DataFrame Mercari.

    Returns
    -------
    df_processed : pd.DataFrame
        DataFrame dengan kolom tambahan:
        brand_name, cat_main, cat_sub1, cat_sub2,
        name_clean, desc_clean, text_features,
        brand_id, cat_main_id, cat_sub1_id, log_price
    le_brand, le_cat_main, le_cat_sub1 : LabelEncoder yang sudah di-fit
    """
    df = df.copy()

    # Step 1: Handle missing values
    df["brand_name"]       = df["brand_name"].fillna("Unknown")
    df["item_description"] = df["item_description"].fillna("No description yet")
    df["category_name"]    = df["category_name"].fillna("Other/Other/Other")
    df["name"]             = df["name"].fillna("")

    # Hapus harga 0 / NaN
    df = df[df["price"].notna() & (df["price"] > 0)].reset_index(drop=True)

    # Step 2: Dekomposisi kategori
    cat_parts = df["category_name"].apply(decompose_category)
    df["cat_main"] = [c[0] for c in cat_parts]
    df["cat_sub1"] = [c[1] for c in cat_parts]
    df["cat_sub2"] = [c[2] for c in cat_parts]

    # Step 3: Pembersihan teks
    df["name_clean"] = df["name"].apply(clean_text)
    df["desc_clean"] = df["item_description"].apply(clean_text)

    # Step 4: Gabungkan fitur teks
    df["text_features"] = (
        df["name_clean"] + " " +
        df["brand_name"].str.lower() + " " +
        df["cat_main"].str.lower() + " " +
        df["cat_sub1"].str.lower() + " " +
        df["desc_clean"]
    )

    # Step 5: Label encoding
    le_brand    = LabelEncoder()
    le_cat_main = LabelEncoder()
    le_cat_sub1 = LabelEncoder()

    df["brand_id"]    = le_brand.fit_transform(df["brand_name"].astype(str))
    df["cat_main_id"] = le_cat_main.fit_transform(df["cat_main"].astype(str))
    df["cat_sub1_id"] = le_cat_sub1.fit_transform(df["cat_sub1"].astype(str))

    # Step 6: Log-transform target
    df["log_price"] = np.log1p(df["price"])

    return df, le_brand, le_cat_main, le_cat_sub1


def preprocess_single_item(
    name: str,
    brand_name: str,
    category_name: str,
    item_condition_id: int,
    shipping: int,
    item_description: str,
    le_brand: LabelEncoder,
    le_cat_main: LabelEncoder,
    le_cat_sub1: LabelEncoder,
) -> dict:
    """
    Preprocessing untuk satu item produk baru (inference time).

    Returns
    -------
    dict dengan kunci:
        text_features, item_condition_id, shipping,
        brand_id, cat_main_id, cat_sub1_id
    """
    brand_name    = brand_name or "Unknown"
    item_description = item_description or "No description yet"
    category_name = category_name or "Other/Other/Other"

    cat_main, cat_sub1, _ = decompose_category(category_name)

    name_clean = clean_text(name)
    desc_clean = clean_text(item_description)
    text_feat  = (
        name_clean + " " +
        brand_name.lower() + " " +
        cat_main.lower() + " " +
        cat_sub1.lower() + " " +
        desc_clean
    )

    # Label encoding — handle unseen labels dengan fallback ke Unknown / Other
    def safe_encode(encoder: LabelEncoder, value: str, fallback: str) -> int:
        val = str(value)
        if val in encoder.classes_:
            return int(encoder.transform([val])[0])
        if fallback in encoder.classes_:
            return int(encoder.transform([fallback])[0])
        return 0  # absolute fallback

    brand_id    = safe_encode(le_brand,    brand_name, "Unknown")
    cat_main_id = safe_encode(le_cat_main, cat_main,   "Other")
    cat_sub1_id = safe_encode(le_cat_sub1, cat_sub1,   "Other")

    return {
        "text_features":     text_feat,
        "item_condition_id": int(item_condition_id),
        "shipping":          int(shipping),
        "brand_id":          brand_id,
        "cat_main_id":       cat_main_id,
        "cat_sub1_id":       cat_sub1_id,
    }


class GraphBuilder:
    """
    Membangun HeteroData graph dari DataFrame yang sudah dipreprocess.

    Digunakan baik saat training (build_full) maupun inference (build_inference).

    Parameters
    ----------
    tfidf_vectorizer : fitted TfidfVectorizer  (max_features=TFIDF_MAX_FEATURES)
    n_brands         : jumlah brand unik (diketahui saat training)
    n_cats           : jumlah kategori unik (diketahui saat training)
    """

    def __init__(self, tfidf_vectorizer, n_brands: int, n_cats: int):
        self.tfidf_vec = tfidf_vectorizer
        self.n_brands  = n_brands
        self.n_cats    = n_cats

        # Precompute node features untuk brand & category (tidak berubah)
        self._brand_feat = self._cyclic_identity(n_brands, 64)
        self._cat_feat   = self._cyclic_identity(n_cats,   32)

    @staticmethod
    def _cyclic_identity(n: int, dim: int) -> np.ndarray:
        """Buat matriks identitas berulang berukuran (n, dim)."""
        base = np.eye(min(n, dim), dtype=np.float32)
        reps = int(np.ceil(n / dim))
        return np.tile(base, (reps, 1))[:n]

    def build_from_df(self, df: pd.DataFrame, device: torch.device = None) -> HeteroData:
        """
        Membangun HeteroData graph dari seluruh DataFrame.
        Digunakan untuk training / offline evaluation.
        """
        device = device or torch.device("cpu")
        N = len(df)

        # Product features: TF-IDF (128) + tabular (2) = 130 dims
        X_tfidf = self.tfidf_vec.transform(df["text_features"]).toarray().astype(np.float32)
        tab = df[TAB_FEATURES].values.astype(np.float32)
        tab = (tab - tab.mean(0)) / (tab.std(0) + 1e-8)
        product_feat = np.concatenate([X_tfidf, tab], axis=1)

        data = HeteroData()
        data["product"].x  = torch.tensor(product_feat, dtype=torch.float)
        data["product"].y  = torch.tensor(df["log_price"].values, dtype=torch.float)
        data["brand"].x    = torch.tensor(self._brand_feat, dtype=torch.float)
        data["category"].x = torch.tensor(self._cat_feat,  dtype=torch.float)

        prod_idx  = torch.arange(N, dtype=torch.long)
        brand_idx = torch.tensor(df["brand_id"].values,    dtype=torch.long)
        cat_idx   = torch.tensor(df["cat_main_id"].values, dtype=torch.long)

        data["product", "has_brand",     "brand"].edge_index    = torch.stack([prod_idx, brand_idx])
        data["product", "in_category",   "category"].edge_index = torch.stack([prod_idx, cat_idx])

        data = ToUndirected()(data)
        return data.to(device)

    def build_inference(
        self,
        item: dict,
        tab_mean: np.ndarray,
        tab_std: np.ndarray,
        device: torch.device = None,
    ) -> HeteroData:
        """
        Membangun graph minimal untuk inference SATU produk.

        Item masuk sebagai node produk ke-0 yang terhubung ke brand & kategorinya.

        Parameters
        ----------
        item     : output dari preprocess_single_item()
        tab_mean : mean tabular saat training  shape (2,)
        tab_std  : std tabular saat training   shape (2,)
        """
        device = device or torch.device("cpu")

        # Product feature
        tfidf_vec = self.tfidf_vec.transform([item["text_features"]]).toarray().astype(np.float32)
        tab = np.array([[item["item_condition_id"], item["shipping"]]], dtype=np.float32)
        tab = (tab - tab_mean) / (tab_std + 1e-8)
        product_feat = np.concatenate([tfidf_vec, tab], axis=1)  # (1, 130)

        data = HeteroData()
        data["product"].x  = torch.tensor(product_feat, dtype=torch.float)
        data["brand"].x    = torch.tensor(self._brand_feat, dtype=torch.float)
        data["category"].x = torch.tensor(self._cat_feat,  dtype=torch.float)

        b_idx = min(item["brand_id"],    self.n_brands - 1)
        c_idx = min(item["cat_main_id"], self.n_cats   - 1)

        data["product", "has_brand",   "brand"].edge_index    = torch.tensor([[0], [b_idx]], dtype=torch.long)
        data["product", "in_category", "category"].edge_index = torch.tensor([[0], [c_idx]], dtype=torch.long)

        data = ToUndirected()(data)
        return data.to(device)
