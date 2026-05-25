export const CATEGORY_OPTIONS = [
  "Women/Tops & Blouses/T-Shirts",
  "Women/Tops & Blouses/Blouse",
  "Women/Dresses/Above Knee",
  "Women/Bottoms/Jeans",
  "Women/Shoes/Athletic",
  "Men/Tops & Shirts/T-Shirts",
  "Men/Pants/Jeans",
  "Men/Shoes/Athletic",
  "Kids/Clothing/Girls",
  "Kids/Clothing/Boys",
  "Beauty/Makeup/Face",
  "Beauty/Skincare/Face",
  "Electronics/Phones & Accessories/Cell Phones",
  "Electronics/Computers & Tablets/Laptops & Computers",
  "Home/Home Décor/Picture Frames & Displays",
  "Sports & Outdoors/Outdoor Recreation/Camping & Hiking",
  "Other/Other/Other",
];

export const CONDITION_LABELS = {
  1: { label: "Baru", color: "#16a34a" },
  2: { label: "Baru — tanpa tag", color: "#65a30d" },
  3: { label: "Baik", color: "#ca8a04" },
  4: { label: "Cukup Baik", color: "#ea580c" },
  5: { label: "Buruk", color: "#dc2626" },
};

export const INITIAL_FORM = {
  name: "",
  brand_name: "",
  category_name: "Other/Other/Other",
  item_condition_id: 1,
  shipping: 0,
  item_description: "",
};

export const MODEL_INFO = [
  { key: "graphsage_price", label: "GraphSAGE", type: "GNN", color: "#4f46e5" },
  { key: "gat_price", label: "GAT", type: "GNN", color: "#7c3aed" },
  { key: "tfidf_ridge_price", label: "TF-IDF + Ridge", type: "Baseline", color: "#0891b2" },
  { key: "xgboost_price", label: "XGBoost", type: "Baseline", color: "#0d9488" },
];

export const USD_TO_IDR = 15500;

export const fmt = (val) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(val);

export const fmtIDR = (val) =>
  new Intl.NumberFormat("id-ID", { style: "currency", currency: "IDR" }).format(val * USD_TO_IDR);

// Shared Styles
export const labelStyle = {
  display: "block",
  fontSize: 13,
  fontWeight: 500,
  color: "#374151",
  marginBottom: 6,
};

export const inputStyle = {
  width: "100%",
  padding: "9px 12px",
  fontSize: 14,
  border: "1px solid #d1d5db",
  borderRadius: 8,
  outline: "none",
  color: "#111827",
  background: "#fff",
  boxSizing: "border-box",
  fontFamily: "inherit",
};

export const primaryBtnStyle = {
  flex: 1,
  padding: "11px 20px",
  fontSize: 14,
  fontWeight: 600,
  color: "#fff",
  background: "#4f46e5",
  border: "none",
  borderRadius: 10,
  cursor: "pointer",
  transition: "background 0.15s",
};

export const secondaryBtnStyle = {
  padding: "11px 20px",
  fontSize: 14,
  fontWeight: 500,
  color: "#374151",
  background: "#f9fafb",
  border: "1px solid #e5e7eb",
  borderRadius: 10,
  cursor: "pointer",
};