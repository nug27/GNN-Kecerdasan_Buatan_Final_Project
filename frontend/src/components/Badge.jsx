export default function Badge({ children, color = "#4f46e5" }) {
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