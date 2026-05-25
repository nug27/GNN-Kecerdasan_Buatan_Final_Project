import SecondPrice from "./components/SecondPrice";

export default function App() {
  return (
    <div style={{ minHeight: "100vh", background: "#f9fafb", padding: "32px 16px" }}>
      <SecondPrice apiUrl={import.meta.env.VITE_API_URL || "http://localhost:8000"} />
    </div>
  );
}
