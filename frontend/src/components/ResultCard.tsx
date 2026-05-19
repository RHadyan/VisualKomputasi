"use client";

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
}

interface ResultCardProps {
  result: PredictionResult;
}

export default function ResultCard({ result }: ResultCardProps) {
  const isReal = result.label === "REAL";
  const confidencePercent = (result.confidence * 100).toFixed(1);

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
      {/* Header with result */}
      <div
        className={`px-6 py-4 ${
          isReal ? "bg-green-50 border-b border-green-200" : "bg-red-50 border-b border-red-200"
        }`}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div
              className={`w-12 h-12 rounded-full flex items-center justify-center ${
                isReal ? "bg-green-100" : "bg-red-100"
              }`}
            >
              {isReal ? (
                <svg className="w-6 h-6 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              ) : (
                <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              )}
            </div>
            <div>
              <h3 className={`text-lg font-bold ${isReal ? "text-green-800" : "text-red-800"}`}>
                Struk {isReal ? "ASLI" : "PALSU"}
              </h3>
              <p className="text-sm text-gray-600">{result.filename}</p>
            </div>
          </div>
          <div className="text-right">
            <p className={`text-2xl font-bold ${isReal ? "text-green-700" : "text-red-700"}`}>
              {confidencePercent}%
            </p>
            <p className="text-xs text-gray-500">Confidence</p>
          </div>
        </div>
      </div>

      {/* Scores detail */}
      <div className="px-6 py-4">
        <h4 className="text-sm font-semibold text-gray-700 mb-3">Detail Skor</h4>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <ScoreItem
            label="Visual Score (CNN)"
            value={result.visual_score}
            description="Skor dari model EfficientNetB0"
          />
          <ScoreItem
            label="Text Score (OCR)"
            value={result.text_score}
            description="Skor validasi teks struk"
          />
          <ScoreItem
            label="Hybrid Score"
            value={result.hybrid_score}
            description="Skor gabungan (0.7*visual + 0.3*text)"
          />
        </div>

        {/* Mode indicator */}
        <div className="mt-4 flex items-center gap-2">
          <span
            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
              result.mode === "real"
                ? "bg-green-100 text-green-800"
                : "bg-yellow-100 text-yellow-800"
            }`}
          >
            {result.mode === "real" ? "Model Aktif" : "Mode Demo (Dummy)"}
          </span>
        </div>
      </div>

      {/* Grad-CAM Heatmap */}
      {result.heatmap && (
        <div className="px-6 py-4 border-t border-gray-100">
          <h4 className="text-sm font-semibold text-gray-700 mb-3">
            Grad-CAM Heatmap
          </h4>
          <p className="text-xs text-gray-500 mb-2">
            Area yang menjadi fokus model dalam mengambil keputusan
          </p>
          <img
            src={`data:image/png;base64,${result.heatmap}`}
            alt="Grad-CAM Heatmap"
            className="w-full max-w-sm rounded-lg shadow-sm border border-gray-200"
          />
        </div>
      )}
    </div>
  );
}

function ScoreItem({
  label,
  value,
  description,
}: {
  label: string;
  value: number | null;
  description: string;
}) {
  return (
    <div className="bg-gray-50 rounded-lg p-3">
      <p className="text-xs text-gray-500">{label}</p>
      <p className="text-lg font-bold text-gray-800">
        {value !== null ? `${(value * 100).toFixed(1)}%` : "N/A"}
      </p>
      <p className="text-xs text-gray-400 mt-1">{description}</p>
    </div>
  );
}
