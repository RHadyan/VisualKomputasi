"use client";

import { useState } from "react";
import UploadForm from "@/components/UploadForm";
import ResultCard from "@/components/ResultCard";

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

export default function Home() {
  const [result, setResult] = useState<PredictionResult | null>(null);

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      {/* Hero section */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          Deteksi Struk Palsu
        </h1>
        <p className="text-gray-600 max-w-2xl mx-auto">
          Upload gambar struk/receipt untuk mendeteksi apakah struk tersebut asli
          atau palsu menggunakan model EfficientNetB0 + OCR Hybrid.
        </p>
      </div>

      {/* Upload section */}
      <div className="mb-8">
        <UploadForm onResult={setResult} />
      </div>

      {/* Result section */}
      {result && (
        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
          <ResultCard result={result} />
        </div>
      )}

      {/* Info section */}
      {!result && (
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
          <InfoCard
            title="Upload Gambar"
            description="Drag & drop atau klik untuk upload gambar struk dalam format JPG, PNG, BMP, atau WebP"
          />
          <InfoCard
            title="Analisis AI"
            description="Model EfficientNetB0 menganalisis visual struk, dilengkapi validasi teks via OCR"
          />
          <InfoCard
            title="Hasil Instan"
            description="Dapatkan hasil deteksi REAL/FAKE beserta confidence score dan LIME heatmap"
          />
        </div>
      )}
    </div>
  );
}

function InfoCard({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <div className="bg-white rounded-xl p-5 border border-gray-200 shadow-sm">
      <h3 className="font-semibold text-gray-800 mb-2">{title}</h3>
      <p className="text-sm text-gray-600">{description}</p>
    </div>
  );
}
