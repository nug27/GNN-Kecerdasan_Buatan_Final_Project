import { useState } from "react";
import { useSecondPrice } from "../hooks/useSecondPrice";
import HealthDot from "./HealthDot";
import EnsembleCard from "./EnsembleCard";
import ModelCard from "./ModelCard";
import {
  CATEGORY_OPTIONS,
  CONDITION_LABELS,
  INITIAL_FORM,
  MODEL_INFO,
  USD_TO_IDR,
  labelStyle,
  inputStyle,
  primaryBtnStyle,
  secondaryBtnStyle,
} from "../utils/constants";

export default function SecondPrice({ apiUrl = "http://localhost:8000" }) {
  const [form, setForm] = useState(INITIAL_FORM);
  const [validationError, setValidationError] = useState(null);

  // Menggunakan custom hook yang sudah Anda buat
  const { predict, loading, error: apiError, result, health, checkHealth, reset } = useSecondPrice(apiUrl);

  const handleChange = (field) => (e) => {
    const val =
      field === "item_condition_id" || field === "shipping"
        ? Number(e.target.value)
        : e.target.value;
    setForm((prev) => ({ ...prev, [field]: val }));
  };

  const handleSubmit = async () => {
    if (!form.name.trim()) {
      setValidationError("Nama produk wajib diisi.");
      return;
    }
    setValidationError(null);
    await predict(form);
  };

  const handleReset = () => {
    setForm(INITIAL_FORM);
    setValidationError(null);
    reset();
  };

  // Gabungkan error dari validasi lokal atau dari API backend
  const displayError = validationError || apiError;

  const topModelKey = result
    ? MODEL_INFO.reduce((best, m) => (result[m.key] > result[best.key] ? m : best), MODEL_INFO[0]).key
    : null;

  return (
    <div style={{ maxWidth: 720, margin: "0 auto", padding: "24px 16px", fontFamily: "system-ui, sans-serif" }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
          <div>
            <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, color: "#111827" }}>
               SecondPrice
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

          <div>
            <label style={labelStyle}>Kategori</label>
            <select value={form.category_name} onChange={handleChange("category_name")} style={inputStyle}>
              {CATEGORY_OPTIONS.map((c) => (
                <option key={c} value={c}>{c}</option>
              ))}
            </select>
          </div>

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

        {displayError && (
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
            ⚠️ {displayError}
          </div>
        )}

        <div style={{ display: "flex", gap: 12, marginTop: 20 }}>
          <button onClick={handleSubmit} disabled={loading} style={primaryBtnStyle}>
            {loading ? <span>⏳ Memproses...</span> : <span> Prediksi Harga</span>}
          </button>
          <button onClick={handleReset} style={secondaryBtnStyle}>
            Reset
          </button>
        </div>
      </div>

      {/* Result */}
      {result && (
        <div>
          <EnsembleCard price={result.ensemble_price} />
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