

import os
import re
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import xgboost as xgb

from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv
from torch_geometric.transforms import ToUndirected


# ============================================================================
# CONFIG & CONSTANTS
# ============================================================================

SEED = 42
MAX_ROWS = 200_000

# Update path ini sesuai lokasi train.tsv
# Contoh: Path('d:/data/train.tsv') atau Path('./train.tsv')
TRAIN_FILE = Path('d:/AI/backend/model/content/train2.csv')  # CSV file path
ARTIFACT_DIR = Path('backend/model')


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed=SEED):
    """Set random seed untuk reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clean_text(value):
    """Bersihkan teks: lowercase, hapus special chars, normalize spaces."""
    value = str(value).lower()
    value = re.sub(r'[^a-z0-9\s]', ' ', value)
    value = re.sub(r'\s+', ' ', value).strip()
    return value


def split_category(value):
    """Pisah category_name menjadi 3 level."""
    parts = str(value).split('/')
    parts = (parts + ['Other', 'Other', 'Other'])[:3]
    return parts[0], parts[1], parts[2]


def fit_mapping(series):
    """Fit LabelEncoder dan buat mapping untuk kategori."""
    encoder = LabelEncoder()
    encoder.fit(series.astype(str).fillna('UNK'))
    mapping = {label: idx for idx, label in enumerate(encoder.classes_)}
    return encoder, mapping


def encode_with_unknown(series, mapping):
    """Encode series dengan handling untuk unseen categories."""
    unknown_index = len(mapping)
    return np.array(
        [mapping.get(str(value), unknown_index) for value in series],
        dtype=np.int64
    )


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class GraphSAGERegressor(nn.Module):
    """Heterogeneous GraphSAGE model untuk price prediction."""

    def __init__(self, in_channels_dict, num_brands, num_cats,
                 hidden_channels=128, dropout=0.3):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout

        self.brand_emb = nn.Embedding(num_brands, hidden_channels)
        self.cat_emb = nn.Embedding(num_cats, hidden_channels)

        self.input_proj = nn.ModuleDict({
            'product': nn.Linear(in_channels_dict['product'], hidden_channels),
            'brand': nn.Linear(hidden_channels, hidden_channels),
            'category': nn.Linear(hidden_channels, hidden_channels),
        })

        self.conv1 = HeteroConv({
            ('product', 'has_brand', 'brand'): SAGEConv((-1, -1), hidden_channels, normalize=True),
            ('brand', 'rev_has_brand', 'product'): SAGEConv((-1, -1), hidden_channels, normalize=True),
            ('product', 'in_category', 'category'): SAGEConv((-1, -1), hidden_channels, normalize=True),
            ('category', 'rev_in_category', 'product'): SAGEConv((-1, -1), hidden_channels, normalize=True),
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('product', 'has_brand', 'brand'): SAGEConv((-1, -1), hidden_channels, normalize=True),
            ('brand', 'rev_has_brand', 'product'): SAGEConv((-1, -1), hidden_channels, normalize=True),
            ('product', 'in_category', 'category'): SAGEConv((-1, -1), hidden_channels, normalize=True),
            ('category', 'rev_in_category', 'product'): SAGEConv((-1, -1), hidden_channels, normalize=True),
        }, aggr='mean')

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, data):
        x_dict = {
            'product': F.relu(self.input_proj['product'](data['product'].x)),
            'brand': F.relu(self.input_proj['brand'](self.brand_emb(data['brand'].node_id))),
            'category': F.relu(self.input_proj['category'](self.cat_emb(data['category'].node_id))),
        }
        x_dict = {key: F.dropout(value, p=self.dropout, training=self.training)
                  for key, value in x_dict.items()}
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {key: F.relu(value) for key, value in x_dict.items()}
        x_dict = {key: F.dropout(value, p=self.dropout, training=self.training)
                  for key, value in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {key: F.relu(value) for key, value in x_dict.items()}
        x_dict = {key: F.dropout(value, p=self.dropout, training=self.training)
                  for key, value in x_dict.items()}
        return self.mlp(x_dict['product']).squeeze(-1)


class GATRegressor(nn.Module):
    """Heterogeneous GAT model untuk price prediction."""

    def __init__(self, in_channels_dict, num_brands, num_cats,
                 hidden_channels=64, dropout=0.3, heads=4):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.dropout = dropout
        self.heads = heads

        self.brand_emb = nn.Embedding(num_brands, hidden_channels)
        self.cat_emb = nn.Embedding(num_cats, hidden_channels)

        self.input_proj = nn.ModuleDict({
            'product': nn.Linear(in_channels_dict['product'], hidden_channels),
            'brand': nn.Linear(hidden_channels, hidden_channels),
            'category': nn.Linear(hidden_channels, hidden_channels),
        })

        self.conv1 = HeteroConv({
            ('product', 'has_brand', 'brand'):
                GATConv((-1, -1), hidden_channels // heads, heads=heads,
                       dropout=dropout, add_self_loops=False),
            ('brand', 'rev_has_brand', 'product'):
                GATConv((-1, -1), hidden_channels // heads, heads=heads,
                       dropout=dropout, add_self_loops=False),
            ('product', 'in_category', 'category'):
                GATConv((-1, -1), hidden_channels // heads, heads=heads,
                       dropout=dropout, add_self_loops=False),
            ('category', 'rev_in_category', 'product'):
                GATConv((-1, -1), hidden_channels // heads, heads=heads,
                       dropout=dropout, add_self_loops=False),
        }, aggr='mean')

        self.conv2 = HeteroConv({
            ('product', 'has_brand', 'brand'):
                GATConv((-1, -1), hidden_channels, heads=1,
                       dropout=dropout, add_self_loops=False),
            ('brand', 'rev_has_brand', 'product'):
                GATConv((-1, -1), hidden_channels, heads=1,
                       dropout=dropout, add_self_loops=False),
            ('product', 'in_category', 'category'):
                GATConv((-1, -1), hidden_channels, heads=1,
                       dropout=dropout, add_self_loops=False),
            ('category', 'rev_in_category', 'product'):
                GATConv((-1, -1), hidden_channels, heads=1,
                       dropout=dropout, add_self_loops=False),
        }, aggr='mean')

        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, 32),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, data):
        x_dict = {
            'product': F.elu(self.input_proj['product'](data['product'].x)),
            'brand': F.elu(self.input_proj['brand'](self.brand_emb(data['brand'].node_id))),
            'category': F.elu(self.input_proj['category'](self.cat_emb(data['category'].node_id))),
        }
        x_dict = {key: F.dropout(value, p=self.dropout, training=self.training)
                  for key, value in x_dict.items()}
        x_dict = self.conv1(x_dict, data.edge_index_dict)
        x_dict = {key: F.elu(value) for key, value in x_dict.items()}
        x_dict = {key: F.dropout(value, p=self.dropout, training=self.training)
                  for key, value in x_dict.items()}
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_dict = {key: F.elu(value) for key, value in x_dict.items()}
        x_dict = {key: F.dropout(value, p=self.dropout, training=self.training)
                  for key, value in x_dict.items()}
        return self.mlp(x_dict['product']).squeeze(-1)


# ============================================================================
# TRAINING & EVALUATION FUNCTIONS
# ============================================================================

def train_gnn(model, data, epochs=60, lr=5e-3, weight_decay=1e-4, log_every=10):
    """Train GNN model dengan early stopping berdasarkan validation loss."""
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    loss_fn = nn.MSELoss()
    best_state = None
    best_val = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        train_loss = loss_fn(
            out[data['product'].train_mask],
            data['product'].y[data['product'].train_mask]
        )
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(data)
            val_loss = loss_fn(
                val_out[data['product'].val_mask],
                data['product'].y[data['product'].val_mask]
            ).item()

        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {key: value.detach().cpu().clone()
                         for key, value in model.state_dict().items()}

        if epoch == 1 or epoch % log_every == 0:
            print(f'Epoch {epoch:03d} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss:.4f}')

    return best_state, best_val


def predict_gnn(model, data, mask):
    """Generate predictions dari GNN model pada subset data."""
    model.eval()
    with torch.no_grad():
        pred = model(data)
    return pred[mask].detach().cpu().numpy()


def compute_regression_metrics(y_true_log, y_pred_log):
    """Compute MAE, RMSE, RMSLE, R2 dari log-scale predictions."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.maximum(y_pred, 0)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {'MAE': mae, 'RMSE': rmse, 'RMSLE': rmsle, 'R2': r2}


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_and_preprocess(filepath, nrows=MAX_ROWS):
    """Load dan preprocess dataset, return train/val/test splits."""
    df = pd.read_csv(filepath, sep=',', nrows=nrows)
    df = df[df['price'] > 0].reset_index(drop=True)

    df['brand_name'] = df['brand_name'].fillna('No Brand')
    df['item_description'] = df['item_description'].fillna('')
    df['category_name'] = df['category_name'].fillna('Other/Other/Other')

    # Split categories
    cat_parts = df['category_name'].apply(split_category)
    df['cat_1'] = cat_parts.apply(lambda x: x[0])
    df['cat_2'] = cat_parts.apply(lambda x: x[1])
    df['cat_3'] = cat_parts.apply(lambda x: x[2])

    # Clean text
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
    df['log_price'] = np.log1p(df['price'])

    # Train/val/test split
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED)

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True)
    )


# ============================================================================
# BASELINE MODELS
# ============================================================================

def train_baseline_models(train_df, val_df, test_df):
    """Train TF-IDF + Ridge dan XGBoost models."""

    # TF-IDF + Ridge
    tfidf_ridge = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)),
        ('ridge', Ridge(alpha=5.0))
    ])
    tfidf_ridge.fit(train_df['text_combined'], train_df['log_price'])
    test_pred_ridge = tfidf_ridge.predict(test_df['text_combined'])

    # XGBoost
    le_brand, brand_mapping = fit_mapping(train_df['brand_name'])
    le_cat_main, cat_main_mapping = fit_mapping(train_df['cat_1'])
    le_cat_sub1, cat_sub1_mapping = fit_mapping(train_df['cat_2'])

    X_train_xgb = np.column_stack([
        train_df['item_condition_id'].values,
        train_df['shipping'].values,
        encode_with_unknown(train_df['brand_name'], brand_mapping),
        encode_with_unknown(train_df['cat_1'], cat_main_mapping)
    ]).astype(np.float32)

    X_val_xgb = np.column_stack([
        val_df['item_condition_id'].values,
        val_df['shipping'].values,
        encode_with_unknown(val_df['brand_name'], brand_mapping),
        encode_with_unknown(val_df['cat_1'], cat_main_mapping)
    ]).astype(np.float32)

    X_test_xgb = np.column_stack([
        test_df['item_condition_id'].values,
        test_df['shipping'].values,
        encode_with_unknown(test_df['brand_name'], brand_mapping),
        encode_with_unknown(test_df['cat_1'], cat_main_mapping)
    ]).astype(np.float32)

    y_train_xgb = train_df['log_price'].values
    y_val_xgb = val_df['log_price'].values

    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        objective='reg:squarederror',
        random_state=SEED,
        n_jobs=-1,
        subsample=0.9,
        colsample_bytree=0.9
    )
    xgb_model.fit(X_train_xgb, y_train_xgb, eval_set=[(X_val_xgb, y_val_xgb)], verbose=False)

    test_pred_xgb = xgb_model.predict(X_test_xgb)

    return {
        'tfidf_ridge': tfidf_ridge,
        'xgb_model': xgb_model,
        'test_pred_ridge': test_pred_ridge,
        'test_pred_xgb': test_pred_xgb,
        'le_brand': le_brand,
        'le_cat_main': le_cat_main,
        'le_cat_sub1': le_cat_sub1,
        'brand_mapping': brand_mapping,
        'cat_main_mapping': cat_main_mapping,
        'cat_sub1_mapping': cat_sub1_mapping,
    }


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_heterogeneous_graph(train_df, val_df, test_df, device):
    """Build heterogeneous graph dengan product, brand, category nodes."""

    # TF-IDF features
    tfidf_vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    tfidf_vec.fit(train_df['text_combined'])

    train_tfidf = tfidf_vec.transform(train_df['text_combined'])
    val_tfidf = tfidf_vec.transform(val_df['text_combined'])
    test_tfidf = tfidf_vec.transform(test_df['text_combined'])

    svd = TruncatedSVD(n_components=128, random_state=SEED)
    train_tfidf_128 = svd.fit_transform(train_tfidf)
    val_tfidf_128 = svd.transform(val_tfidf)
    test_tfidf_128 = svd.transform(test_tfidf)

    def build_product_features(df_part, tfidf_128):
        return np.hstack([
            tfidf_128,
            df_part['item_condition_id'].values.reshape(-1, 1),
            df_part['shipping'].values.reshape(-1, 1)
        ]).astype(np.float32)

    prod_train = build_product_features(train_df, train_tfidf_128)
    prod_val = build_product_features(val_df, val_tfidf_128)
    prod_test = build_product_features(test_df, test_tfidf_128)

    scaler = StandardScaler()
    prod_train = scaler.fit_transform(prod_train)
    prod_val = scaler.transform(prod_val)
    prod_test = scaler.transform(prod_test)

    prod_features = np.vstack([prod_train, prod_val, prod_test]).astype(np.float32)
    full_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)

    # Brand dan category mappings
    brand_encoder, brand_mapping = fit_mapping(train_df['brand_name'])
    cat_main_encoder, cat_main_mapping = fit_mapping(train_df['cat_1'])

    def encode_graph_nodes(series, mapping):
        unknown_index = len(mapping)
        return np.array(
            [mapping.get(str(value), unknown_index) for value in series],
            dtype=np.int64
        )

    brand_ids_all = np.concatenate([
        encode_graph_nodes(train_df['brand_name'], brand_mapping),
        encode_graph_nodes(val_df['brand_name'], brand_mapping),
        encode_graph_nodes(test_df['brand_name'], brand_mapping)
    ])
    cat_ids_all = np.concatenate([
        encode_graph_nodes(train_df['cat_1'], cat_main_mapping),
        encode_graph_nodes(val_df['cat_1'], cat_main_mapping),
        encode_graph_nodes(test_df['cat_1'], cat_main_mapping)
    ])

    num_products = len(full_df)
    num_brands = len(brand_mapping) + 1
    num_cats = len(cat_main_mapping) + 1

    # Build HeteroData
    data = HeteroData()
    data['product'].x = torch.tensor(prod_features, dtype=torch.float32)
    data['product'].y = torch.tensor(full_df['log_price'].values, dtype=torch.float32)
    data['brand'].node_id = torch.arange(num_brands, dtype=torch.long)
    data['category'].node_id = torch.arange(num_cats, dtype=torch.long)

    src_index = torch.arange(num_products, dtype=torch.long)
    brand_edge_index = torch.stack([src_index, torch.tensor(brand_ids_all, dtype=torch.long)])
    cat_edge_index = torch.stack([src_index, torch.tensor(cat_ids_all, dtype=torch.long)])

    data['product', 'has_brand', 'brand'].edge_index = brand_edge_index
    data['product', 'in_category', 'category'].edge_index = cat_edge_index
    data = ToUndirected()(data)

    # Train/val/test masks
    train_mask = torch.zeros(num_products, dtype=torch.bool)
    val_mask = torch.zeros(num_products, dtype=torch.bool)
    test_mask = torch.zeros(num_products, dtype=torch.bool)

    train_mask[:len(train_df)] = True
    val_mask[len(train_df):len(train_df) + len(val_df)] = True
    test_mask[len(train_df) + len(val_df):] = True

    data['product'].train_mask = train_mask
    data['product'].val_mask = val_mask
    data['product'].test_mask = test_mask
    data = data.to(device)

    in_channels_dict = {'product': data['product'].x.size(1)}

    return data, tfidf_vec, num_brands, num_cats, in_channels_dict


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(train_file=TRAIN_FILE, artifact_dir=ARTIFACT_DIR):
    """Run full end-to-end pipeline."""

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Load dan preprocess
    print('\n--- Loading and Preprocessing Data ---')
    train_df, val_df, test_df = load_and_preprocess(train_file, nrows=MAX_ROWS)
    print(f'Train / Val / Test: {len(train_df):,} / {len(val_df):,} / {len(test_df):,}')

    # Baseline models
    print('\n--- Training Baseline Models ---')
    baseline_result = train_baseline_models(train_df, val_df, test_df)
    print('✅ TF-IDF + Ridge and XGBoost trained.')

    # Build graph
    print('\n--- Building Heterogeneous Graph ---')
    data, tfidf_vec, num_brands, num_cats, in_channels_dict = build_heterogeneous_graph(
        train_df, val_df, test_df, device
    )
    print(f'Product feature dimension: {in_channels_dict["product"]}')
    print(f'Brands / Categories: {num_brands} / {num_cats}')

    # Train GNN models
    print('\n--- Training GraphSAGE ---')
    sage_model = GraphSAGERegressor(in_channels_dict, num_brands, num_cats,
                                   hidden_channels=128, dropout=0.3).to(device)
    best_state_sage, best_val_sage = train_gnn(sage_model, data, epochs=60)
    sage_model.load_state_dict(best_state_sage)
    test_pred_sage = predict_gnn(sage_model, data, data['product'].test_mask)
    print(f'Best validation loss (GraphSAGE): {best_val_sage:.6f}')

    print('\n--- Training GAT ---')
    gat_model = GATRegressor(in_channels_dict, num_brands, num_cats,
                            hidden_channels=64, dropout=0.3, heads=4).to(device)
    best_state_gat, best_val_gat = train_gnn(gat_model, data, epochs=60)
    gat_model.load_state_dict(best_state_gat)
    test_pred_gat = predict_gnn(gat_model, data, data['product'].test_mask)
    print(f'Best validation loss (GAT): {best_val_gat:.6f}')

    # Evaluate
    print('\n--- Evaluation ---')
    y_test_log = test_df['log_price'].values
    metrics = []

    for model_name, pred in [
        ('TF-IDF + Ridge', baseline_result['test_pred_ridge']),
        ('XGBoost', baseline_result['test_pred_xgb']),
        ('GraphSAGE', test_pred_sage),
        ('GAT', test_pred_gat),
    ]:
        result = compute_regression_metrics(y_test_log, pred)
        result['Model'] = model_name
        metrics.append(result)

    results_df = pd.DataFrame(metrics).sort_values('RMSLE').reset_index(drop=True)
    print(results_df)

    # Save artifacts
    print('\n--- Saving Artifacts ---')
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    torch.save(sage_model.state_dict(), artifact_dir / 'graphsage_secondprice.pt')
    torch.save(gat_model.state_dict(), artifact_dir / 'gat_secondprice.pt')

    with open(artifact_dir / 'tfidf_ridge_pipeline.pkl', 'wb') as f:
        pickle.dump(baseline_result['tfidf_ridge'], f)

    with open(artifact_dir / 'xgboost_model.pkl', 'wb') as f:
        pickle.dump(baseline_result['xgb_model'], f)

    preprocessor_meta = {
        'le_brand': baseline_result['le_brand'],
        'le_cat_main': baseline_result['le_cat_main'],
        'le_cat_sub1': baseline_result['le_cat_sub1'],
        'tfidf_vec': tfidf_vec,
        'in_channels_dict': in_channels_dict,
        'n_brands': num_brands,
        'n_cats': num_cats,
        'SEED': SEED,
        'columns': ['name', 'brand_name', 'category_name', 'item_condition_id',
                   'item_description', 'shipping'],
        'brand_mapping': baseline_result['brand_mapping'],
        'cat_main_mapping': baseline_result['cat_main_mapping'],
        'brand_unknown_index': num_brands - 1,
        'cat_unknown_index': num_cats - 1,
        'tfidf_max_features': 50000,
        'graph_feature_dim': 130,
        'max_rows': MAX_ROWS
    }

    with open(artifact_dir / 'preprocessor_meta.pkl', 'wb') as f:
        pickle.dump(preprocessor_meta, f)

    print(f'✅ Artifacts saved to {artifact_dir}')

    return {
        'results_df': results_df,
        'models': {
            'sage': sage_model,
            'gat': gat_model,
            'tfidf_ridge': baseline_result['tfidf_ridge'],
            'xgb': baseline_result['xgb_model'],
        },
        'data': data,
        'metadata': preprocessor_meta
    }


if __name__ == "__main__":
    run_pipeline()
