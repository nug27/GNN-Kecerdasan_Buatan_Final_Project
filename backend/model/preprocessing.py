

import re
import pandas as pd
import numpy as np
from typing import Tuple


def clean_text(text: str) -> str:
    """Bersihkan teks: lowercase, hapus special chars, normalize spaces."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def split_category(value):
    """Pisah category_name menjadi 3 level.

    Contoh: "Women/Tops & Blouses/T-Shirts" → ("Women", "Tops & Blouses", "T-Shirts")
    """
    parts = str(value).split('/')
    parts = (parts + ['Other', 'Other', 'Other'])[:3]
    return parts[0], parts[1], parts[2]


def load_data(filepath: str, nrows: int = 200_000) -> pd.DataFrame:
    """Load dataset dengan sample limit."""
    df = pd.read_csv(filepath, sep='\t', nrows=nrows)
    return df


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Jalankan full preprocessing pipeline pada dataframe.

    Steps:
    1. Hapus harga = 0 (tidak valid)
    2. Isi nilai kosong
    3. Pisah kategori menjadi 3 level
    4. Bersihkan teks
    5. Gabung teks
    6. Tambah log_price
    """
    df = df.copy()

    # Remove invalid prices
    df = df[df['price'] > 0].reset_index(drop=True)

    # Fill missing values
    df['brand_name'] = df['brand_name'].fillna('No Brand')
    df['item_description'] = df['item_description'].fillna('')
    df['category_name'] = df['category_name'].fillna('Other/Other/Other')

    # Split category menjadi 3 level
    cat_parts = df['category_name'].apply(split_category)
    df['cat_1'] = cat_parts.apply(lambda x: x[0])
    df['cat_2'] = cat_parts.apply(lambda x: x[1])
    df['cat_3'] = cat_parts.apply(lambda x: x[2])

    # Clean text columns
    df['name_clean'] = df['name'].fillna('').apply(clean_text)
    df['brand_clean'] = df['brand_name'].apply(clean_text)
    df['cat1_clean'] = df['cat_1'].apply(clean_text)
    df['desc_clean'] = df['item_description'].apply(clean_text)
    df['text_combined'] = (
        df['name_clean'] + ' ' +
        df['brand_clean'] + ' ' +
        df['cat1_clean'] + ' ' +
        df['desc_clean']
    )

    # Add log_price untuk normalisasi distribusi harga
    df['log_price'] = np.log1p(df['price'])

    return df


def select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Pilih kolom relevan untuk tahap selanjutnya."""
    final_cols = [
        'train_id',
        'name',
        'name_clean',
        'item_condition_id',
        'brand_name',
        'cat_1', 'cat_2', 'cat_3',
        'shipping',
        'text_combined',
        'price',
        'log_price'
    ]
    return df[final_cols].copy()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Load dan preprocess
    df = load_data('train.tsv', nrows=200_000)
    df = preprocess_dataset(df)
    df_clean = select_columns(df)

    # Save hasil
    output_path = 'train_clean.csv'
    df_clean.to_csv(output_path, index=False)

    print(f'✅ Dataset preprocessed: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns')
    print(f'✅ Saved to {output_path}')