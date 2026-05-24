/**
 * SecondPrice — React UI Component
 * Terhubung ke FastAPI backend di secondprice_backend/app.py
 *
 * Cara pakai:
 *   import SecondPrice from './components/SecondPrice'
 *   <SecondPrice apiUrl="http://localhost:8000" />
 *
 * Props:
 *   apiUrl  : string  — URL backend FastAPI (default: http://localhost:8000)
 */

import { useState, useCallback } from "react";

// ── Konstanta ────────────────────────────────────────────────────────────────

const CATEGORY_OPTIONS = [
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

const CONDITION_LABELS = {
  1: { label: "Baru", color: "#16a34a" },
  2: { label: "Baru — tanpa tag", color: "#65a30d" },
  3: { label: "Baik", color: "#ca8a04" },
  4: { label: "Cukup Baik", color: "#ea580c" },
  5: { label: "Buruk", color: "#dc2626" },
};

const INITIAL_FORM = {
  name: "",
  brand_name: "",
  category_name: "Other/Other/Other",
  item_condition_id: 1,
  shipping: 0,
  item_description: "",
};

const MODEL_INFO = [
  { key: "graphsage_price", label: "GraphSAGE", type: "GNN", color: "#4f46e5" },
  { key: "gat_price",       label: "GAT",       type: "GNN", color: "#7c3aed" },
  { key: "tfidf_ridge_price", label: "TF-IDF + Ridge", type: "Baseline", color: "#0891b2" },
  { key: "xgboost_price",   label: "XGBoost",   type: "Baseline", color: "#0d9488" },
];

// ── Helpers ──────────────────────────────────────────────────────────────────

const USD_TO_IDR = 15500; // Kurs konversi (update sesuai kurs real)

const fmt = (val) =>
  new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(val);

const fmtIDR = (val) =>
  new Intl.NumberFormat("id-ID", { style: "currency", currency: "IDR" }).format(val * USD_TO_IDR);

// ── Sub-components ────────────────────────────────────────────────────────────

function Badge({ children, color = "#4f46e5" }) {
  return (
    <span
      style={{
        fontSize: 11,
        fontWeight: 500,
        padding: "2px 8px",
        borderRadius: 99,
        backgroundColor: color + "18",
        color,
        border: `1px solid ${color}30`,
        letterSpacing: "0.02em",
      }}
    >
      {children}
    </span>
  );
}

function ModelCard({ label, type, price, color, isTop }) {
  return (
    <div
      style={{
        background: "#fff",
        border: isTop ? `2px solid ${color}` : "1px solid #e5e7eb",
        borderRadius: 12,
        padding: "16px 20px",
        position: "relative",
        transition: "box-shadow 0.15s",
      }}
    >
      {isTop && (
        <span
          style={{
            position: "absolute",
            top: -11,
            left: 16,
            background: color,
            color: "#fff",
            fontSize: 10,
            fontWeight: 600,
            padding: "2px 10px",
            borderRadius: 99,
            letterSpacing: "0.05em",
          }}
        >
          TERBAIK
        </span>
      )}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
        <div>
          <p style={{ margin: 0, fontSize: 13, color: "#6b7280", fontWeight: 500 }}>{label}</p>
          <Badge color={color}>{type}</Badge>
        </div>
        <div style={{ textAlign: "right" }}>
          <p style={{ margin: 0, fontSize: 20, fontWeight: 700, color }}>{fmt(price)}</p>
          <p style={{ margin: "4px 0 0", fontSize: 11, color: "#9ca3af" }}>{fmtIDR(price)}</p>
        </div>
      </div>
    </div>
  );
}

function EnsembleCard({ price }) {
  return (
    <div
      style={{
        background: "linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)",
        borderRadius: 16,
        padding: "20px 24px",
        color: "#fff",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}
    >
       <div>
        <p style={{ margin: 0, fontSize: 13, opacity: 0.85, fontWeight: 500 }}>
          Rekomendasi Harga Jual
        </p>
        <p style={{ margin: "4px 0 0", fontSize: 11, opacity: 0.65 }}>
          Median dari 4 model (robust, tidak sensitif outlier)
        </p>
      </div>
      <div style={{ textAlign: "right" }}>
        <p style={{ margin: 0, fontSize: 32, fontWeight: 800 }}>{fmt(price)}</p>
        <p style={{ margin: "4px 0 0", fontSize: 12, opacity: 0.85 }}>{fmtIDR(price)}</p>
      </div>
    </div>
  );
}

function HealthDot({ status }) {
  const ok = status === "ok";
  return (
    <span
      style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: "#6b7280" }}
    >
      <span
        style={{
          width: 8,
          height: 8,
          borderRadius: "50%",
          background: ok ? "#16a34a" : status === null ? "#d1d5db" : "#dc2626",
          display: "inline-block",
        }}
      />
      {ok ? "Backend terhubung" : status === null ? "Mengecek..." : "Backend tidak tersedia"}
    </span>
  );
}

// ── Main Component ────────────────────────────────────────────────────────────

export default function SecondPrice({ apiUrl = "http://localhost:8000" }) {
  const [form, setForm] = useState(INITIAL_FORM);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState(null); // null | "ok" | "error"

  // Cek koneksi backend saat mount
  const checkHealth = useCallback(async () => {
    setHealth(null);
    try {
      const res = await fetch(`${apiUrl}/health`);
      const data = await res.json();
      setHealth(data.status === "ok" ? "ok" : "error");
    } catch {
      setHealth("error");
    }
  }, [apiUrl]);

  // Jalankan health check sekali
  useState(() => { checkHealth(); }, []);

  const handleChange = (field) => (e) => {
    const val =
      field === "item_condition_id" || field === "shipping"
        ? Number(e.target.value)
        : e.target.value;
    setForm((prev) => ({ ...prev, [field]: val }));
  };

  const handleSubmit = async () => {
    if (!form.name.trim()) {
      setError("Nama produk wajib diisi.");
      return;
    }
    setError(null);
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch(`${apiUrl}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Gagal mendapat prediksi.");
      }
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setForm(INITIAL_FORM);
    setResult(null);
    setError(null);
  };

  // Cari model dengan harga tertinggi/terendah untuk highlight
  const topModelKey = result
    ? MODEL_INFO.reduce((best, m) =>
        result[m.key] > result[best.key] ? m : best
      , MODEL_INFO[0]).key
    : null;

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div style={{ maxWidth: 720, margin: "0 auto", padding: "24px 16px", fontFamily: "system-ui, sans-serif" }}>

      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: "#111827" }}>
              🏷️ SecondPrice
            </h1>
            <p style={{ margin: "4px 0 0", fontSize: 14, color: "#6b7280" }}>
              Prediksi harga barang bekas dengan Graph Neural Network
            </p>
          </div>
          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 6 }}>
            <HealthDot status={health} />
            <button
              onClick={checkHealth}
              style={{ fontSize: 11, color: "#6b7280", background: "none", border: "none", cursor: "pointer", padding: 0 }}
            >
              cek ulang
            </button>
          </div>
        </div>
      </div>

      {/* Form Card */}
      <div
        style={{
          background: "#fff",
          border: "1px solid #e5e7eb",
          borderRadius: 16,
          padding: "24px",
          marginBottom: 24,
        }}
      >
        <h2 style={{ margin: "0 0 20px", fontSize: 15, fontWeight: 600, color: "#374151" }}>
          Detail Produk
        </h2>

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>

          {/* Nama Produk */}
          <div style={{ gridColumn: "1 / -1" }}>
            <label style={labelStyle}>Nama Produk *</label>
            <input
              type="text"
              placeholder="mis. Nike Air Max 90 White Size 10"
              value={form.name}
              onChange={handleChange("name")}
              style={inputStyle}
            />
          </div>

          {/* Brand */}
          <div>
            <label style={labelStyle}>Merek (Brand)</label>
            <input
              type="text"
              placeholder="mis. Nike, Zara, Samsung..."
              value={form.brand_name}
              onChange={handleChange("brand_name")}
              style={inputStyle}
            />
          </div>

          {/* Kategori */}
          <div>
            <label style={labelStyle}>Kategori</label>
            <select
              value={form.category_name}
              onChange={handleChange("category_name")}
              style={inputStyle}
            >
              {CATEGORY_OPTIONS.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>

          {/* Kondisi */}
          <div>
            <label style={labelStyle}>
              Kondisi Barang &nbsp;
              <span style={{ fontWeight: 600, color: CONDITION_LABELS[form.item_condition_id].color }}>
                — {CONDITION_LABELS[form.item_condition_id].label}
              </span>
            </label>
            <input
              type="range"
              min={1} max={5} step={1}
              value={form.item_condition_id}
              onChange={handleChange("item_condition_id")}
              style={{ width: "100%", accentColor: CONDITION_LABELS[form.item_condition_id].color }}
            />
            <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, color: "#9ca3af", marginTop: 2 }}>
              <span>1 — Baru</span><span>5 — Buruk</span>
            </div>
          </div>

          {/* Ongkir */}
          <div>
            <label style={labelStyle}>Siapa yang bayar ongkir?</label>
            <div style={{ display: "flex", gap: 12, marginTop: 8 }}>
              {[
                { val: 0, label: "Pembeli" },
                { val: 1, label: "Penjual (inkl. harga)" },
              ].map(({ val, label }) => (
                <label
                  key={val}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    fontSize: 14,
                    cursor: "pointer",
                    padding: "8px 16px",
                    borderRadius: 8,
                    border: form.shipping === val ? "2px solid #4f46e5" : "1px solid #e5e7eb",
                    background: form.shipping === val ? "#eef2ff" : "#fff",
                    color: form.shipping === val ? "#4f46e5" : "#374151",
                    fontWeight: form.shipping === val ? 600 : 400,
                    flex: 1,
                    transition: "all 0.1s",
                  }}
                >
                  <input
                    type="radio"
                    name="shipping"
                    value={val}
                    checked={form.shipping === val}
                    onChange={handleChange("shipping")}
                    style={{ display: "none" }}
                  />
                  {label}
                </label>
              ))}
            </div>
          </div>

          {/* Deskripsi */}
          <div style={{ gridColumn: "1 / -1" }}>
            <label style={labelStyle}>Deskripsi Produk</label>
            <textarea
              rows={3}
              placeholder="Deskripsikan kondisi, ukuran, warna, kelengkapan, dll..."
              value={form.item_description}
              onChange={handleChange("item_description")}
              style={{ ...inputStyle, resize: "vertical", lineHeight: 1.6 }}
            />
          </div>
        </div>

        {/* Error */}
        {error && (
          <div
            style={{
              marginTop: 16,
              padding: "10px 14px",
              borderRadius: 8,
              background: "#fef2f2",
              border: "1px solid #fecaca",
              color: "#b91c1c",
              fontSize: 13,
            }}
          >
            ⚠️ {error}
          </div>
        )}

        {/* Actions */}
        <div style={{ display: "flex", gap: 12, marginTop: 20 }}>
          <button onClick={handleSubmit} disabled={loading} style={primaryBtnStyle}>
            {loading ? (
              <span>⏳ Memproses...</span>
            ) : (
              <span>🔮 Prediksi Harga</span>
            )}
          </button>
          <button onClick={handleReset} style={secondaryBtnStyle}>
            Reset
          </button>
        </div>
      </div>

      {/* Result */}
      {result && (
        <div>
          {/* Ensemble Highlight */}
          <EnsembleCard price={result.ensemble_price} />

          {/* Per-model breakdown */}
          <div style={{ marginTop: 16, marginBottom: 8 }}>
            <p style={{ margin: 0, fontSize: 13, color: "#6b7280", fontWeight: 500 }}>
              Rincian per model
            </p>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            {MODEL_INFO.map((m) => (
              <ModelCard
                key={m.key}
                label={m.label}
                type={m.type}
                price={result[m.key]}
                color={m.color}
                isTop={m.key === topModelKey}
              />
            ))}
          </div>

           <p style={{ margin: "16px 0 0", fontSize: 12, color: "#9ca3af", textAlign: "center" }}>
             Harga dalam USD & IDR (kurs: 1 USD = Rp {USD_TO_IDR.toLocaleString("id-ID")}) · Model: GraphSAGE, GAT, TF-IDF+Ridge, XGBoost · Hanya estimasi.
           </p>
        </div>
      )}
    </div>
  );
}

// ── Shared Styles ─────────────────────────────────────────────────────────────

const labelStyle = {
  display: "block",
  fontSize: 13,
  fontWeight: 500,
  color: "#374151",
  marginBottom: 6,
};

const inputStyle = {
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

const primaryBtnStyle = {
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

const secondaryBtnStyle = {
  padding: "11px 20px",
  fontSize: 14,
  fontWeight: 500,
  color: "#374151",
  background: "#f9fafb",
  border: "1px solid #e5e7eb",
  borderRadius: 10,
  cursor: "pointer",
};
