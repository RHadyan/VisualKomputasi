"use client";

import { useState, useCallback } from "react";

interface PredictionResult {
  id: number;
  filename: string;
  label: string;
  confidence: number;
  visual_score: number | null;
  text_score: number | null;
  hybrid_score: number | null;
  heatmap: string | null;
  image_url: string;
  mode: string;
  zona_stats: { header: number; isi: number; footer: number } | null;
  explanation: {
    alasan: string[];
    saran: string;
    zona_aktif: string[];
    zona_scores: { header: number; isi: number; footer: number };
    active_ratio: number;
    confidence_level: string;
  } | null;
}

interface UploadFormProps {
  onResult: (result: PredictionResult) => void;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function UploadForm({ onResult }: UploadFormProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const handleFile = useCallback(
    async (file: File) => {
      setError(null);

      // Validate file type
      const allowedTypes = [
        "image/jpeg",
        "image/png",
        "image/bmp",
        "image/webp",
      ];
      if (!allowedTypes.includes(file.type)) {
        setError("Format file tidak didukung. Gunakan JPG, PNG, BMP, atau WebP.");
        return;
      }

      // Validate file size (10MB)
      if (file.size > 10 * 1024 * 1024) {
        setError("Ukuran file terlalu besar. Maksimal 10MB.");
        return;
      }

      // Show preview
      const reader = new FileReader();
      reader.onload = (e) => setPreview(e.target?.result as string);
      reader.readAsDataURL(file);

      // Upload to API
      setIsLoading(true);
      try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(`${API_URL}/api/predict`, {
          method: "POST",
          body: formData,
        });

        if (!response.ok) {
          const data = await response.json();
          throw new Error(data.detail || "Prediction failed");
        }

        const result: PredictionResult = await response.json();
        onResult(result);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "Terjadi kesalahan saat memproses gambar"
        );
      } finally {
        setIsLoading(false);
      }
    },
    [onResult]
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setIsDragging(false);
  }, []);

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  return (
    <div className="w-full">
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        className={`relative border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer ${
          isDragging
            ? "border-blue-500 bg-blue-50"
            : "border-gray-300 hover:border-blue-400 hover:bg-gray-50"
        }`}
      >
        <input
          type="file"
          accept="image/jpeg,image/png,image/bmp,image/webp"
          onChange={handleInputChange}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          disabled={isLoading}
        />

        {isLoading ? (
          <div className="flex flex-col items-center gap-3">
            <div className="w-10 h-10 border-4 border-blue-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-gray-600 font-medium">Menganalisis struk...</p>
          </div>
        ) : preview ? (
          <div className="flex flex-col items-center gap-3">
            <img
              src={preview}
              alt="Preview"
              className="max-h-48 rounded-lg shadow-sm"
            />
            <p className="text-sm text-gray-500">
              Klik atau drag gambar lain untuk mengganti
            </p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-3">
            <svg
              className="w-12 h-12 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={1.5}
                d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
              />
            </svg>
            <div>
              <p className="text-gray-700 font-medium">
                Drag & drop gambar struk di sini
              </p>
              <p className="text-sm text-gray-500 mt-1">
                atau klik untuk memilih file (JPG, PNG, BMP, WebP - maks 10MB)
              </p>
            </div>
          </div>
        )}
      </div>

      {error && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-red-700 text-sm">{error}</p>
        </div>
      )}
    </div>
  );
}
