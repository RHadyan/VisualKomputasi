import re
import io
import numpy as np
from PIL import Image

_reader = None


def get_reader():
    global _reader
    if _reader is None:
        try:
            import easyocr
            _reader = easyocr.Reader(["en", "id"], gpu=False)
            print("[OCR] EasyOCR reader loaded successfully")
        except Exception as e:
            print(f"[OCR] Failed to load EasyOCR: {e}")
            return None
    return _reader


def extract_text(image_bytes: bytes) -> str:
    reader = get_reader()
    if reader is None:
        return ""

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_array = np.array(img)

    results = reader.readtext(img_array)
    text = " ".join([result[1] for result in results])
    return text


def text_validation(text: str) -> float:
    """
    Persis Colab text_validation():
    Skor mulai 1.0, dikurangi per fitur struk yang tidak ditemukan.
    """
    score = 1.0
    text_upper = text.upper()

    # Cek kata kunci total (EN + ID)
    keywords_total = ['TOTAL', 'JUMLAH', 'SUBTOTAL', 'GRAND TOTAL', 'AMOUNT']
    if not any(k in text_upper for k in keywords_total):
        score -= 0.25

    # Cek jumlah angka (harga barang)
    numbers = re.findall(r'\d+', text)
    if len(numbers) < 3:
        score -= 0.20

    # Cek format harga (12.000 atau 12,500)
    harga_pattern = re.findall(r'\d{1,3}[.,]\d{3}', text)
    if not harga_pattern:
        score -= 0.20

    # Cek format tanggal (DD/MM/YYYY atau DD-MM-YYYY)
    tanggal_pattern = re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text)
    if not tanggal_pattern:
        score -= 0.20

    # Cek nama toko / header struk (huruf kapital beruntun)
    nama_toko_pattern = re.findall(r'\b[A-Z]{3,}\b', text)
    if not nama_toko_pattern:
        score -= 0.15

    return max(score, 0.0)
