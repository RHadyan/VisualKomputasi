import numpy as np
import os
import glob
import cv2
from PIL import Image, ImageOps
import io

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
_SUPPORTED_EXTS = (".h5", ".keras")
_model = None
_use_dummy = True


def _find_model_file() -> str | None:
    preferred = os.path.join(MODELS_DIR, "model_ku.keras")
    if os.path.isfile(preferred):
        return preferred
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


def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image, pts):
    rect = _order_points(pts)
    tl, tr, br, bl = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def crop_struk(image_bytes: bytes) -> Image.Image:
    """
    Auto-detect dan crop area struk dari background menggunakan kontur + filter warna terang.
    """
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")

    original = img_bgr.copy()
    ratio = img_bgr.shape[0] / 900.0
    resized = cv2.resize(img_bgr, (int(img_bgr.shape[1] / ratio), 900))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    kernel = np.ones((7, 7), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    resized_area = resized.shape[0] * resized.shape[1]

    receipt_contour = None
    for contour in contours[:10]:
        area = cv2.contourArea(contour)
        if area < 0.1 * resized_area:
            continue
        if area > 0.95 * resized_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            receipt_contour = approx
            break
        approx = cv2.approxPolyDP(contour, 0.05 * peri, True)
        if len(approx) == 4:
            receipt_contour = approx
            break
        x, y, w, h = cv2.boundingRect(contour)
        receipt_contour = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32).reshape(4, 1, 2)
        break

    if receipt_contour is None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    pts = receipt_contour.reshape(4, 2).astype(np.float32) * ratio
    warped = _four_point_transform(original, pts)
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    return Image.fromarray(warped_rgb)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((224, 224), Image.LANCZOS)

    delta_w = 224 - img.size[0]
    delta_h = 224 - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    img = ImageOps.expand(img, padding, fill=(255, 255, 255))

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

    if visual_score > 0.65:
        label = "REAL"
        confidence = visual_score
        text_score = None
        hybrid_score = None
    elif visual_score < 0.35:
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
    img_array = preprocess_image(image_bytes)
    prediction = _model.predict(img_array, verbose=0)
    raw_score = float(prediction[0][0])

    # raw_score tinggi = REAL, rendah = FAKE
    visual_score = raw_score

    if visual_score > 0.65:
        label = "REAL"
        confidence = visual_score
        text_score = None
        hybrid_score = None
    elif visual_score < 0.35:
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

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
        "visual_score": round(float(visual_score), 4),
        "text_score": round(float(text_score), 4) if text_score is not None else None,
        "hybrid_score": round(float(hybrid_score), 4) if hybrid_score is not None else None,
        "mode": "real",
    }


def predict(image_bytes: bytes) -> dict:
    if _use_dummy:
        return predict_dummy(image_bytes)
    else:
        return predict_real(image_bytes)
