/**
 * useSecondPrice — Custom React Hook
 * Abstraksi semua komunikasi dengan SecondPrice FastAPI backend.
 *
 * Cara pakai:
 *   const { predict, predictBatch, health, checkHealth } = useSecondPrice("http://localhost:8000")
 */

import { useState, useCallback, useEffect } from "react";

export function useSecondPrice(apiUrl = "http://localhost:8000") {

  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(null);
  const [result, setResult]     = useState(null);
  const [health, setHealth]     = useState(null); // null | "ok" | "error"
  const [models, setModels]     = useState([]);

  // ── Health check ──────────────────────────────────────────────────────────

  const checkHealth = useCallback(async () => {
    setHealth(null);
    try {
      const res  = await fetch(`${apiUrl}/health`);
      const data = await res.json();
      setHealth(data.status === "ok" ? "ok" : "error");
      return data;
    } catch {
      setHealth("error");
      return null;
    }
  }, [apiUrl]);

  // ── Daftar model ──────────────────────────────────────────────────────────

  const fetchModels = useCallback(async () => {
    try {
      const res  = await fetch(`${apiUrl}/models`);
      const data = await res.json();
      setModels(data.models || []);
    } catch {
      setModels([]);
    }
  }, [apiUrl]);

  // Cek health & fetch models saat hook pertama kali dipakai
  useEffect(() => {
    checkHealth();
    fetchModels();
  }, [checkHealth, fetchModels]);

  // ── Prediksi satu produk ──────────────────────────────────────────────────

  /**
   * predict(item) → PredictResponse | null
   *
   * item: {
   *   name, brand_name?, category_name?,
   *   item_condition_id?, shipping?, item_description?
   * }
   */
  const predict = useCallback(async (item) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${apiUrl}/predict`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify(item),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
      return data;
    } catch (e) {
      setError(e.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, [apiUrl]);

  // ── Prediksi batch ────────────────────────────────────────────────────────

  /**
   * predictBatch(items) → BatchPredictResponse | null
   *
   * items: array of item objects (maks. 100)
   */
  const predictBatch = useCallback(async (items) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await fetch(`${apiUrl}/predict/batch`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ items }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setResult(data);
      return data;
    } catch (e) {
      setError(e.message);
      return null;
    } finally {
      setLoading(false);
    }
  }, [apiUrl]);

  // ── Reset state ───────────────────────────────────────────────────────────

  const reset = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return {
    // State
    loading,
    error,
    result,
    health,
    models,

    // Actions
    predict,
    predictBatch,
    checkHealth,
    fetchModels,
    reset,
  };
}
