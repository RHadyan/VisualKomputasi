import numpy as np
import os
import glob
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


def predict_dummy(image_bytes: bytes) -> dict:
    """Dummy prediction for testing without a real model."""
    # Generate a pseudo-random prediction based on image data
    img_data = np.frombuffer(image_bytes[:1000], dtype=np.uint8)
    seed = int(img_data.sum()) % 100

    # Simulate different confidence levels
    if seed < 40:
        visual_score = 0.15 + (seed / 100) * 0.3
    elif seed < 70:
        visual_score = 0.4 + (seed / 100) * 0.3
    else:
        visual_score = 0.7 + (seed / 100) * 0.25

    visual_score = min(max(visual_score, 0.0), 1.0)

    # Simulate hybrid logic — sama kayak Colab threshold 0.45-0.55
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
    img_array = preprocess_image(image_bytes)
    prediction = _model.predict(img_array, verbose=0)
    raw_score = float(prediction[0][0])

    visual_score = raw_score

    # Zona ragu 0.45-0.55 — persis Colab
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

        # Fallback ke CNN jika OCR gagal baca (< 10 karakter)
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
    """Main prediction function. Uses real model if available, otherwise dummy."""
    if _use_dummy:
        return predict_dummy(image_bytes)
    else:
        return predict_real(image_bytes)
