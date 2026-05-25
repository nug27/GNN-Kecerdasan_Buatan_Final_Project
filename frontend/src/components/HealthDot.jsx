export default function HealthDot({ status }) {
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