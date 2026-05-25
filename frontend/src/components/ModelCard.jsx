import Badge from "./Badge";
import { fmt, fmtIDR } from "../utils/constants";

export default function ModelCard({ label, type, price, color, isTop }) {
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