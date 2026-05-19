"use client";

import { useState, useEffect } from "react";

interface Prediction {
  id: number;
  filename: string;
  label: string;
  confidence: number;
  visual_score: number | null;
  text_score: number | null;
  hybrid_score: number | null;
  image_path: string | null;
  created_at: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function HistoryPage() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchHistory = async () => {
    try {
      setLoading(true);
      const response = await fetch(`${API_URL}/api/history`);
      if (!response.ok) throw new Error("Failed to fetch history");
      const data = await response.json();
      setPredictions(data.predictions);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Gagal memuat history"
      );
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: number) => {
    try {
      const response = await fetch(`${API_URL}/api/history/${id}`, {
        method: "DELETE",
      });
      if (!response.ok) throw new Error("Failed to delete");
      setPredictions((prev) => prev.filter((p) => p.id !== id));
    } catch (err) {
      console.error("Delete failed:", err);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">
          History Prediksi
        </h1>
        <div className="flex items-center justify-center py-12">
          <div className="w-8 h-8 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">
          History Prediksi
        </h1>
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-700">{error}</p>
          <p className="text-sm text-red-500 mt-1">
            Pastikan backend server berjalan di {API_URL}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-3xl font-bold text-gray-900">History Prediksi</h1>
        <span className="text-sm text-gray-500">
          {predictions.length} hasil
        </span>
      </div>

      {predictions.length === 0 ? (
        <div className="text-center py-12 bg-white rounded-xl border border-gray-200">
          <svg
            className="w-16 h-16 text-gray-300 mx-auto mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <p className="text-gray-500">Belum ada history prediksi</p>
          <p className="text-sm text-gray-400 mt-1">
            Upload gambar struk di halaman Deteksi untuk memulai
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {predictions.map((prediction) => (
            <HistoryItem
              key={prediction.id}
              prediction={prediction}
              onDelete={handleDelete}
            />
          ))}
        </div>
      )}
    </div>
  );
}

function HistoryItem({
  prediction,
  onDelete,
}: {
  prediction: Prediction;
  onDelete: (id: number) => void;
}) {
  const isReal = prediction.label === "REAL";
  const date = new Date(prediction.created_at);
  const formattedDate = date.toLocaleDateString("id-ID", {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-4 flex items-center justify-between hover:shadow-sm transition-shadow">
      <div className="flex items-center gap-4">
        {/* Status indicator */}
        <div
          className={`w-10 h-10 rounded-full flex items-center justify-center ${
            isReal ? "bg-green-100" : "bg-red-100"
          }`}
        >
          {isReal ? (
            <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          ) : (
            <svg className="w-5 h-5 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          )}
        </div>

        {/* Info */}
        <div>
          <p className="font-medium text-gray-800">{prediction.filename}</p>
          <div className="flex items-center gap-3 mt-1">
            <span
              className={`text-sm font-semibold ${
                isReal ? "text-green-700" : "text-red-700"
              }`}
            >
              {isReal ? "ASLI" : "PALSU"}
            </span>
            <span className="text-sm text-gray-500">
              {(prediction.confidence * 100).toFixed(1)}% confidence
            </span>
            <span className="text-xs text-gray-400">{formattedDate}</span>
          </div>
        </div>
      </div>

      {/* Delete button */}
      <button
        onClick={() => onDelete(prediction.id)}
        className="p-2 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-lg transition-colors"
        title="Hapus"
      >
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
          />
        </svg>
      </button>
    </div>
  );
}
