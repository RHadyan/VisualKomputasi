"use client";

import { useState, useEffect } from "react";

interface ModelInfo {
  model_name: string;
  architecture: string;
  input_size: string;
  output: string;
  preprocessing: string;
  hybrid_logic: {
    description: string;
    formula: string;
  };
  ocr_validation: {
    description: string;
    checks: string[];
  };
  training: {
    optimizer: string;
    loss: string;
    epochs: string;
    dataset_split: string;
  };
  is_dummy_mode: boolean;
  note: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function AboutPage() {
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchModelInfo = async () => {
      try {
        const response = await fetch(`${API_URL}/api/model-info`);
        if (response.ok) {
          const data = await response.json();
          setModelInfo(data);
        }
      } catch {
        // Use fallback data if API is not available
      } finally {
        setLoading(false);
      }
    };
    fetchModelInfo();
  }, []);

  return (
    <div className="max-w-4xl mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold text-gray-900 mb-2">Tentang Model</h1>
      <p className="text-gray-600 mb-8">
        Informasi teknis tentang model deteksi struk palsu
      </p>

      {/* Status */}
      {modelInfo && (
        <div
          className={`mb-6 p-4 rounded-lg border ${
            modelInfo.is_dummy_mode
              ? "bg-yellow-50 border-yellow-200"
              : "bg-green-50 border-green-200"
          }`}
        >
          <p
            className={`text-sm font-medium ${
              modelInfo.is_dummy_mode ? "text-yellow-800" : "text-green-800"
            }`}
          >
            {modelInfo.note}
          </p>
        </div>
      )}

      <div className="space-y-6">
        {/* Architecture */}
        <Section title="Arsitektur Model">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <InfoItem label="Model" value="EfficientNetB0 (Transfer Learning)" />
            <InfoItem label="Input" value="224 x 224 x 3 (RGB)" />
            <InfoItem label="Output" value="Binary Classification (REAL / FAKE)" />
            <InfoItem label="Preprocessing" value="EfficientNet preprocess_input" />
          </div>
          <div className="mt-4 bg-gray-900 rounded-lg p-4 overflow-x-auto">
            <pre className="text-sm text-green-400 font-mono">
{`EfficientNetB0 (pretrained ImageNet, frozen)
    -> GlobalAveragePooling2D
    -> Dense(128, activation='relu')
    -> Dropout(0.5)
    -> Dense(1, activation='sigmoid')`}
            </pre>
          </div>
        </Section>

        {/* Hybrid Logic */}
        <Section title="Hybrid CNN + OCR Logic">
          <p className="text-gray-600 mb-4">
            Model menggunakan pendekatan hybrid yang menggabungkan analisis visual
            (CNN) dengan validasi teks (OCR) untuk meningkatkan akurasi.
          </p>
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
            <h4 className="font-semibold text-blue-800 mb-2">Alur Keputusan:</h4>
            <ol className="list-decimal list-inside space-y-2 text-sm text-blue-900">
              <li>
                <strong>Visual score &gt; 0.55</strong> → Langsung prediksi{" "}
                <span className="text-green-700 font-bold">REAL</span> (CNN yakin)
              </li>
              <li>
                <strong>Visual score &lt; 0.45</strong> → Langsung prediksi{" "}
                <span className="text-red-700 font-bold">FAKE</span> (CNN yakin)
              </li>
              <li>
                <strong>Visual score 0.45 - 0.55</strong> → CNN tidak yakin, panggil OCR untuk validasi teks
              </li>
            </ol>
            <div className="mt-3 p-3 bg-white rounded border border-blue-100">
              <p className="text-sm font-mono text-gray-700">
                hybrid_score = 0.7 * visual_score + 0.3 * text_score
              </p>
              <p className="text-xs text-gray-500 mt-1">
                Jika hybrid_score &gt; 0.5 → REAL, selainnya → FAKE
              </p>
            </div>
          </div>
        </Section>

        {/* OCR Validation */}
        <Section title="Validasi Teks (OCR)">
          <p className="text-gray-600 mb-4">
            EasyOCR digunakan untuk membaca teks dari gambar struk dan memvalidasi
            kontennya. OCR hanya dipanggil jika CNN tidak yakin (visual score 0.45 - 0.55).
          </p>
          <div className="space-y-2">
            <CheckItem
              text='Cek keberadaan kata kunci total: TOTAL, JUMLAH, SUBTOTAL, GRAND TOTAL, AMOUNT (-0.25 jika tidak ditemukan)'
              isPositive={false}
            />
            <CheckItem
              text="Cek minimal 3 angka ditemukan (-0.20 jika kurang)"
              isPositive={false}
            />
            <CheckItem
              text="Cek format harga (12.000 atau 12,500) (-0.20 jika tidak ditemukan)"
              isPositive={false}
            />
            <CheckItem
              text="Cek format tanggal DD/MM/YYYY atau DD-MM-YYYY (-0.20 jika tidak ditemukan)"
              isPositive={false}
            />
            <CheckItem
              text="Cek nama toko / huruf kapital beruntun (-0.15 jika tidak ditemukan)"
              isPositive={false}
            />
          </div>
          <p className="text-sm text-gray-500 mt-3">
            Skor awal: 1.0, dikurangi berdasarkan pengecekan di atas. Minimum: 0.0
          </p>
        </Section>

        {/* Training Config */}
        <Section title="Konfigurasi Training">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <InfoItem label="Optimizer" value="Adam (learning rate: 1e-4)" />
            <InfoItem label="Loss Function" value="Binary Crossentropy" />
            <InfoItem label="Epochs" value="100 (EarlyStopping)" />
            <InfoItem label="Dataset Split" value="70% train / 15% val / 15% test" />
            <InfoItem label="Preprocessing" value="Thumbnail + White Padding (224x224) + EfficientNet preprocess_input" />
            <InfoItem label="Model File" value="model_ku.keras" />
          </div>
        </Section>

        {/* LIME Visualization */}
        <Section title="LIME Visualization">
          <p className="text-gray-600 mb-4">
            LIME (Local Interpretable Model-agnostic Explanations) digunakan untuk
            memvisualisasikan area mana pada gambar struk yang paling berpengaruh
            terhadap keputusan model.
          </p>
          <div className="bg-gray-50 rounded-lg p-4 border border-gray-200">
            <ul className="space-y-2 text-sm text-gray-700">
              <li>
                • Membuat 300 perturbasi (variasi) dari gambar input
              </li>
              <li>
                • Menganalisis superpixel mana yang paling berpengaruh terhadap prediksi
              </li>
              <li>
                • <span className="text-green-600 font-medium">Hijau</span> = area yang mendukung prediksi model
              </li>
              <li>
                • <span className="text-red-600 font-medium">Merah</span> = area yang melemahkan/kontradiksi prediksi
              </li>
              <li>
                • Zona analisis: Header (1/3 atas), Isi Struk (1/3 tengah), Footer (1/3 bawah)
              </li>
            </ul>
          </div>
        </Section>

        {/* Dataset */}
        <Section title="Dataset">
          <p className="text-gray-600 mb-4">
            Dataset terdiri dari gambar struk/receipt yang dikategorikan menjadi dua
            kelas:
          </p>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 text-center">
              <p className="text-2xl font-bold text-green-700">REAL</p>
              <p className="text-sm text-green-600">Struk asli/genuine</p>
            </div>
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-center">
              <p className="text-2xl font-bold text-red-700">FAKE</p>
              <p className="text-sm text-red-600">Struk palsu/forged</p>
            </div>
          </div>
          <p className="text-sm text-gray-500 mt-3">
            Format gambar yang didukung: JPG, JPEG, PNG, BMP, WebP
          </p>
        </Section>
      </div>
    </div>
  );
}

function Section({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
      <h2 className="text-xl font-bold text-gray-800 mb-4">{title}</h2>
      {children}
    </div>
  );
}

function InfoItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-gray-50 rounded-lg p-3">
      <p className="text-xs text-gray-500 uppercase tracking-wide">{label}</p>
      <p className="text-sm font-medium text-gray-800 mt-1">{value}</p>
    </div>
  );
}

function CheckItem({
  text,
  isPositive,
}: {
  text: string;
  isPositive: boolean;
}) {
  return (
    <div className="flex items-start gap-2 p-2 rounded bg-gray-50">
      <span className={isPositive ? "text-green-500" : "text-orange-500"}>
        {isPositive ? "+" : "-"}
      </span>
      <p className="text-sm text-gray-700">{text}</p>
    </div>
  );
}
