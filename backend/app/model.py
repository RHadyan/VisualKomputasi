import numpy as np
import os
import glob
import cv2
from PIL import Image, ImageOps
import io

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_SUPPORTED_EXTS = (".keras", ".h5")
_model = None
_use_dummy = True


def _find_model_file() -> str | None:
    for ext in _SUPPORTED_EXTS:
        matches = sorted(glob.glob(os.path.join(MODELS_DIR, f"*{ext}")))
        if matches:
            return matches[0]
    return None


def _build_model():
    import keras
    model = keras.Sequential([
        keras.applications.EfficientNetB0(
            input_shape=(224, 224, 3),
            include_top=False,
            weights="imagenet",
        ),
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.build((None, 224, 224, 3))
    return model


def load_model():
    global _model, _use_dummy
    import tensorflow as tf

    model_path = _find_model_file()

    if model_path:
        try:
            _model = tf.keras.models.load_model(model_path)
            _model(np.zeros((1, 224, 224, 3), dtype=np.float32))
            _use_dummy = False
            print(f"[MODEL] Loaded real model from {model_path}")
        except Exception as e:
            print(f"[MODEL] Failed to load model with load_model(): {e}")
            print("[MODEL] Trying build + load_weights fallback...")
            try:
                _model = _build_model()
                _model.load_weights(model_path)
                _model(np.zeros((1, 224, 224, 3), dtype=np.float32))
                _use_dummy = False
                print(f"[MODEL] Loaded via build+load_weights from {model_path}")
            except Exception as e2:
                print(f"[MODEL] Fallback also failed: {e2}")
                print("[MODEL] Using dummy prediction mode")
                _use_dummy = True
    else:
        print(f"[MODEL] No model file (*.keras or *.h5) found in {MODELS_DIR}")
        print("[MODEL] Using dummy prediction mode")
        _use_dummy = True


def crop_struk(image_bytes: bytes) -> Image.Image:
    """
    Auto-detect dan crop area struk dari background menggunakan kontur + filter warna putih.
    Persis seperti Colab cell 42.
    """
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    img_h, img_w = img_bgr.shape[:2]
    img_area = img_h * img_w

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    mask_putih = (
        (r.astype(int) > 170) &
        (g.astype(int) > 170) &
        (b.astype(int) > 170) &
        (np.abs(r.astype(int) - g.astype(int)) < 40) &
        (np.abs(g.astype(int) - b.astype(int)) < 40) &
        (np.abs(r.astype(int) - b.astype(int)) < 40)
    ).astype(np.uint8) * 255

    kandidat = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.10 * img_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if not (0.2 <= w / (h + 1e-8) <= 2.0):
            continue
        mask_cnt = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(mask_cnt, [cnt], -1, 255, thickness=cv2.FILLED)
        piksel_dalam = np.sum(mask_cnt == 255)
        piksel_putih = np.sum((mask_putih == 255) & (mask_cnt == 255))
        rasio_putih = piksel_putih / (piksel_dalam + 1e-8)
        if rasio_putih < 0.40:
            continue
        kandidat.append((area, x, y, w, h, rasio_putih))

    if not kandidat:
        return Image.fromarray(img_rgb)

    best = max(kandidat, key=lambda c: c[5])
    _, x, y, w, h, _ = best

    pad = 10
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, img_w)
    y2 = min(y + h + pad, img_h)

    return Image.fromarray(img_rgb[y1:y2, x1:x2])


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image for model input — matches Colab flow_from_directory behavior:
    1. Resize (stretch) to 224x224 (same as target_size in flow_from_directory)
    2. Apply EfficientNet preprocess_input
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)

    img_array = np.array(img, dtype=np.float32)

    if not _use_dummy:
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_array = preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def preprocess_image_with_crop(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess dengan crop_struk dulu (match Colab show_lime cell 42):
    1. Auto-crop area struk
    2. Resize (stretch) to 224x224
    3. Apply EfficientNet preprocess_input
    """
    cropped = crop_struk(image_bytes)
    cropped = cropped.resize((224, 224), Image.LANCZOS)

    img_array = np.array(cropped, dtype=np.float32)

    if not _use_dummy:
        from tensorflow.keras.applications.efficientnet import preprocess_input
        img_array = preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def compute_zona_stats(image_bytes: bytes) -> dict:
    """
    Hitung statistik zona struk (Header/Isi/Footer) berdasarkan Grad-CAM heatmap.
    Return dict dengan skor per zona.
    """
    if _use_dummy or _model is None:
        return {
            "header": 0.12,
            "isi": 0.08,
            "footer": 0.10,
            "active_ratio": 0.15,
        }

    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    input_array = preprocess_input(np.expand_dims(img_array.copy(), axis=0))

    base_model = _model.get_layer("efficientnetb0")
    conv_output = base_model.get_layer("top_conv").output
    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[conv_output, base_model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_output, base_out = grad_model(input_array)
        x = base_out
        for layer in _model.layers[1:]:
            x = layer(x, training=False)
        class_channel = x[:, 0]

    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = last_conv_output[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224))
    ) / 255.0

    h = 224
    header_zone = heatmap_resized[:h // 3, :]
    isi_zone = heatmap_resized[h // 3:2 * h // 3, :]
    footer_zone = heatmap_resized[2 * h // 3:, :]

    active_ratio = float(np.sum(heatmap_resized > 0.3) / heatmap_resized.size)

    return {
        "header": round(float(np.mean(header_zone)), 4),
        "isi": round(float(np.mean(isi_zone)), 4),
        "footer": round(float(np.mean(footer_zone)), 4),
        "active_ratio": round(active_ratio, 4),
    }


def generate_explanation(label: str, zona_stats: dict) -> dict:
    """
    Generate penjelasan alasan prediksi berdasarkan zona stats.
    Persis seperti Colab cell 42 logic.
    """
    header_score = zona_stats["header"]
    isi_score = zona_stats["isi"]
    footer_score = zona_stats["footer"]
    active_ratio = zona_stats["active_ratio"]

    zona_aktif = []
    if header_score > 0.05:
        zona_aktif.append("Header / Nama Toko")
    if isi_score > 0.05:
        zona_aktif.append("Isi Struk (Item)")
    if footer_score > 0.05:
        zona_aktif.append("Footer / Total")

    if not zona_aktif:
        max_zona = max(zona_stats, key=lambda k: zona_stats[k] if k != "active_ratio" else 0)
        zona_map = {"header": "Header / Nama Toko", "isi": "Isi Struk (Item)", "footer": "Footer / Total"}
        zona_aktif = [zona_map.get(max_zona, "Isi Struk (Item)")]

    zona_desc = ", ".join(zona_aktif)

    alasan = []
    if label == "FAKE":
        alasan.append(f"Model menemukan kejanggalan visual di: {zona_desc}.")
        if isi_score > 0.1:
            alasan.append("Bagian item/harga terlihat tidak konsisten — kemungkinan ada yang diedit.")
        if header_score > 0.1:
            alasan.append("Nama toko atau header struk terlihat mencurigakan.")
        if footer_score > 0.1:
            alasan.append("Bagian total/kembali menunjukkan angka yang tidak wajar.")
        if active_ratio > 0.4:
            alasan.append("Area kejanggalan sangat luas — kemungkinan besar hasil manipulasi.")
        elif active_ratio < 0.1:
            alasan.append("Hanya sebagian kecil yang mencurigakan — kemungkinan manipulasi terlokalisir.")
        saran = "Disarankan periksa keaslian struk secara manual."
    else:
        alasan.append(f"Model menemukan ciri khas struk asli di: {zona_desc}.")
        if isi_score > 0.1:
            alasan.append("Bagian item dan harga konsisten dengan format struk asli.")
        if header_score > 0.1:
            alasan.append("Nama toko dan header memiliki format yang sesuai standar.")
        if footer_score > 0.1:
            alasan.append("Bagian total dan kembalian memiliki angka yang wajar dan konsisten.")
        if active_ratio > 0.4:
            alasan.append("Sebagian besar area struk konsisten dengan karakteristik struk asli.")
        saran = "Struk terlihat asli berdasarkan analisis visual model."

    confidence_val = max(header_score, isi_score, footer_score)
    if confidence_val >= 0.15:
        conf_label = "Sangat Yakin"
    elif confidence_val >= 0.05:
        conf_label = "Cukup Yakin"
    else:
        conf_label = "Kurang Yakin"

    return {
        "alasan": alasan,
        "saran": saran,
        "zona_aktif": zona_aktif,
        "zona_scores": {
            "header": header_score,
            "isi": isi_score,
            "footer": footer_score,
        },
        "active_ratio": active_ratio,
        "confidence_level": conf_label,
    }


def predict_dummy(image_bytes: bytes) -> dict:
    img_data = np.frombuffer(image_bytes[:1000], dtype=np.uint8)
    seed = int(img_data.sum()) % 100

    if seed < 40:
        visual_score = 0.15 + (seed / 100) * 0.3
    elif seed < 70:
        visual_score = 0.4 + (seed / 100) * 0.3
    else:
        visual_score = 0.7 + (seed / 100) * 0.25

    visual_score = min(max(visual_score, 0.0), 1.0)

    if visual_score > 0.55:
        label = "REAL"
        confidence = visual_score
        text_score = None
        hybrid_score = None
    elif visual_score < 0.45:
        label = "FAKE"
        confidence = 1 - visual_score
        text_score = None
        hybrid_score = None
    else:
        text_score = 0.5 + (seed % 30) / 100
        hybrid_score = 0.7 * visual_score + 0.3 * text_score
        label = "REAL" if hybrid_score > 0.5 else "FAKE"
        confidence = hybrid_score if label == "REAL" else 1 - hybrid_score

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "visual_score": round(float(visual_score), 4),
        "text_score": round(float(text_score), 4) if text_score is not None else None,
        "hybrid_score": round(float(hybrid_score), 4) if hybrid_score is not None else None,
        "mode": "dummy",
    }


def predict_real(image_bytes: bytes) -> dict:
    img_array = preprocess_image_with_crop(image_bytes)
    prediction = _model.predict(img_array, verbose=0)
    raw_score = float(prediction[0][0])

    # Swap: score tinggi = FAKE, score rendah = REAL
    visual_score = 1 - raw_score

    if visual_score > 0.55:
        label = "REAL"
        confidence = visual_score
        text_score = None
        hybrid_score = None
    elif visual_score < 0.45:
        label = "FAKE"
        confidence = 1 - visual_score
        text_score = None
        hybrid_score = None
    else:
        from .ocr import extract_text, text_validation
        text = extract_text(image_bytes)

        if len(text.strip()) < 10:
            label = "REAL" if visual_score > 0.5 else "FAKE"
            confidence = visual_score if label == "REAL" else 1 - visual_score
            text_score = None
            hybrid_score = None
        else:
            text_score = text_validation(text)
            hybrid_score = 0.7 * visual_score + 0.3 * text_score
            label = "REAL" if hybrid_score > 0.5 else "FAKE"
            confidence = hybrid_score if label == "REAL" else 1 - hybrid_score

    zona_stats = compute_zona_stats(image_bytes)
    explanation = generate_explanation(label, zona_stats)

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "visual_score": round(float(visual_score), 4),
        "text_score": round(float(text_score), 4) if text_score is not None else None,
        "hybrid_score": round(float(hybrid_score), 4) if hybrid_score is not None else None,
        "zona_stats": zona_stats,
        "explanation": explanation,
        "mode": "real",
    }


def predict(image_bytes: bytes) -> dict:
    if _use_dummy:
        return predict_dummy(image_bytes)
    else:
        return predict_real(image_bytes)
