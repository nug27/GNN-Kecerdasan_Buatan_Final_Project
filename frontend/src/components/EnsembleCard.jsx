import { fmt, fmtIDR } from "../utils/constants";

export default function EnsembleCard({ price }) {
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