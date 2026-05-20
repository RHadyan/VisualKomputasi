import numpy as np
import io
import base64
from PIL import Image, ImageOps


def generate_lime_dummy(image_bytes: bytes) -> dict:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)

    seed = int(np.array(img).sum()) % 100
    np.random.seed(seed)
    mask = np.random.choice([0, 1], size=(224, 224), p=[0.7, 0.3])

    lime_display = img_array.copy()
    lime_display[mask == 1, 0] = lime_display[mask == 1, 0] * 0.3
    lime_display[mask == 1, 1] = np.minimum(lime_display[mask == 1, 1] * 0.5 + 150, 255)
    lime_display[mask == 1, 2] = lime_display[mask == 1, 2] * 0.3

    overlay_img = Image.fromarray(lime_display.astype(np.uint8))
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")
    buffer.seek(0)

    heatmap_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return {
        "heatmap": heatmap_b64,
        "zona_stats": {"header": 0.1, "isi": 0.15, "footer": 0.08},
        "explanation": None,
    }


def generate_lime_real(image_bytes: bytes, label: str) -> dict:
    from lime import lime_image
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from .model import _model, crop_struk

    if _model is None:
        return generate_lime_dummy(image_bytes)

    img = crop_struk(image_bytes)
    img = img.resize((224, 224), Image.LANCZOS)
    img_display = np.array(img, dtype=np.float64)

    def predict_fn(images):
        processed = preprocess_input(images.astype("float32").copy())
        probs_real = _model.predict(processed, verbose=0)
        return np.hstack([1 - probs_real, probs_real])

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_display,
        predict_fn,
        top_labels=2,
        hide_color=255,
        num_samples=300,
    )

    pred_probs = predict_fn(np.expand_dims(img_display, 0))[0]
    pred_class = 1 if pred_probs[1] > 0.5 else 0

    _, mask_pos = explanation.get_image_and_mask(
        pred_class, positive_only=True, num_features=10, hide_rest=False
    )
    _, mask_all = explanation.get_image_and_mask(
        pred_class, positive_only=False, num_features=10, hide_rest=False
    )
    mask_neg = ((mask_all == 1) & (mask_pos == 0)).astype(int)

    lime_display = img_display.copy()

    lime_display[mask_pos == 1, 0] = lime_display[mask_pos == 1, 0] * 0.3
    lime_display[mask_pos == 1, 1] = np.minimum(lime_display[mask_pos == 1, 1] * 0.5 + 150, 255)
    lime_display[mask_pos == 1, 2] = lime_display[mask_pos == 1, 2] * 0.3

    lime_display[mask_neg == 1, 0] = np.minimum(lime_display[mask_neg == 1, 0] * 0.5 + 100, 255)
    lime_display[mask_neg == 1, 1] = lime_display[mask_neg == 1, 1] * 0.5
    lime_display[mask_neg == 1, 2] = lime_display[mask_neg == 1, 2] * 0.5

    overlay_img = Image.fromarray(lime_display.astype(np.uint8))
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")
    buffer.seek(0)
    heatmap_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    h, w = mask_pos.shape
    total_px = mask_pos.size
    active_px = int(np.sum(mask_pos > 0))

    zona = {
        "header": mask_pos[:h // 3, :],
        "isi": mask_pos[h // 3:2 * h // 3, :],
        "footer": mask_pos[2 * h // 3:, :],
    }
    skor_zona = {k: round(float(np.mean(v)), 4) for k, v in zona.items()}

    zona_aktif = []
    if skor_zona["header"] > 0.05:
        zona_aktif.append("Header / Nama Toko")
    if skor_zona["isi"] > 0.05:
        zona_aktif.append("Isi Struk (Item)")
    if skor_zona["footer"] > 0.05:
        zona_aktif.append("Footer / Total")

    if not zona_aktif:
        max_zona = max(skor_zona, key=skor_zona.get)
        zona_map = {"header": "Header / Nama Toko", "isi": "Isi Struk (Item)", "footer": "Footer / Total"}
        zona_aktif = [zona_map.get(max_zona, "Isi Struk (Item)")]

    zona_desc = ", ".join(zona_aktif)
    active_ratio = active_px / total_px

    alasan = []
    if label == "FAKE":
        alasan.append(f"Model menemukan kejanggalan visual di: {zona_desc}.")
        if skor_zona["isi"] > 0.1:
            alasan.append("Bagian item/harga terlihat tidak konsisten — kemungkinan ada yang diedit.")
        if skor_zona["header"] > 0.1:
            alasan.append("Nama toko atau header struk terlihat mencurigakan.")
        if skor_zona["footer"] > 0.1:
            alasan.append("Bagian total/kembali menunjukkan angka yang tidak wajar.")
        if active_ratio > 0.4:
            alasan.append("Area kejanggalan sangat luas — kemungkinan besar hasil manipulasi.")
        elif active_ratio < 0.1:
            alasan.append("Hanya sebagian kecil yang mencurigakan — kemungkinan manipulasi terlokalisir.")
        saran = "Disarankan periksa keaslian struk secara manual."
    else:
        alasan.append(f"Model menemukan ciri khas struk asli di: {zona_desc}.")
        if skor_zona["isi"] > 0.1:
            alasan.append("Bagian item dan harga konsisten dengan format struk asli.")
        if skor_zona["header"] > 0.1:
            alasan.append("Nama toko dan header memiliki format yang sesuai standar.")
        if skor_zona["footer"] > 0.1:
            alasan.append("Bagian total dan kembalian memiliki angka yang wajar dan konsisten.")
        if active_ratio > 0.4:
            alasan.append("Sebagian besar area struk konsisten dengan karakteristik struk asli.")
        saran = "Struk terlihat asli berdasarkan analisis visual model."

    confidence_val = max(skor_zona.values())
    if confidence_val >= 0.15:
        conf_label = "Sangat Yakin"
    elif confidence_val >= 0.05:
        conf_label = "Cukup Yakin"
    else:
        conf_label = "Kurang Yakin"

    explanation_data = {
        "alasan": alasan,
        "saran": saran,
        "zona_aktif": zona_aktif,
        "zona_scores": skor_zona,
        "active_ratio": round(active_ratio, 4),
        "confidence_level": conf_label,
    }

    return {
        "heatmap": heatmap_b64,
        "zona_stats": skor_zona,
        "explanation": explanation_data,
    }


def generate_gradcam_real(image_bytes: bytes, label: str) -> dict:
    import tensorflow as tf
    from .model import _model

    if _model is None:
        return generate_lime_dummy(image_bytes)

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224), Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)

    from tensorflow.keras.applications.efficientnet import preprocess_input
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

    heatmap_colored = np.zeros((224, 224, 3), dtype=np.float32)
    heatmap_colored[:, :, 0] = heatmap_resized
    heatmap_colored[:, :, 1] = heatmap_resized * 0.5

    alpha = 0.4
    overlay = (1 - alpha) * (img_array / 255.0) + alpha * heatmap_colored
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    overlay_img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")
    buffer.seek(0)
    heatmap_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    h = 224
    header_zone = heatmap_resized[:h // 3, :]
    isi_zone = heatmap_resized[h // 3:2 * h // 3, :]
    footer_zone = heatmap_resized[2 * h // 3:, :]

    skor_zona = {
        "header": round(float(np.mean(header_zone)), 4),
        "isi": round(float(np.mean(isi_zone)), 4),
        "footer": round(float(np.mean(footer_zone)), 4),
    }

    active_ratio = float(np.sum(heatmap_resized > 0.3) / heatmap_resized.size)

    zona_aktif = []
    if skor_zona["header"] > 0.05:
        zona_aktif.append("Header / Nama Toko")
    if skor_zona["isi"] > 0.05:
        zona_aktif.append("Isi Struk (Item)")
    if skor_zona["footer"] > 0.05:
        zona_aktif.append("Footer / Total")

    if not zona_aktif:
        max_zona = max(skor_zona, key=skor_zona.get)
        zona_map = {"header": "Header / Nama Toko", "isi": "Isi Struk (Item)", "footer": "Footer / Total"}
        zona_aktif = [zona_map.get(max_zona, "Isi Struk (Item)")]

    zona_desc = ", ".join(zona_aktif)

    alasan = []
    if label == "FAKE":
        alasan.append(f"Model menemukan kejanggalan visual di: {zona_desc}.")
        if skor_zona["isi"] > 0.1:
            alasan.append("Bagian item/harga terlihat tidak konsisten — kemungkinan ada yang diedit.")
        if skor_zona["header"] > 0.1:
            alasan.append("Nama toko atau header struk terlihat mencurigakan.")
        if skor_zona["footer"] > 0.1:
            alasan.append("Bagian total/kembali menunjukkan angka yang tidak wajar.")
        saran = "Disarankan periksa keaslian struk secara manual."
    else:
        alasan.append(f"Model menemukan ciri khas struk asli di: {zona_desc}.")
        if skor_zona["isi"] > 0.1:
            alasan.append("Bagian item dan harga konsisten dengan format struk asli.")
        if skor_zona["header"] > 0.1:
            alasan.append("Nama toko dan header memiliki format yang sesuai standar.")
        if skor_zona["footer"] > 0.1:
            alasan.append("Bagian total dan kembalian memiliki angka yang wajar dan konsisten.")
        saran = "Struk terlihat asli berdasarkan analisis visual model."

    confidence_val = max(skor_zona.values())
    if confidence_val >= 0.15:
        conf_label = "Sangat Yakin"
    elif confidence_val >= 0.05:
        conf_label = "Cukup Yakin"
    else:
        conf_label = "Kurang Yakin"

    explanation_data = {
        "alasan": alasan,
        "saran": saran,
        "zona_aktif": zona_aktif,
        "zona_scores": skor_zona,
        "active_ratio": round(active_ratio, 4),
        "confidence_level": conf_label,
    }

    return {
        "heatmap": heatmap_b64,
        "zona_stats": skor_zona,
        "explanation": explanation_data,
    }


def generate_gradcam(image_bytes: bytes, label: str = "REAL") -> dict:
    from .model import _use_dummy

    if _use_dummy:
        return generate_lime_dummy(image_bytes)
    else:
        return generate_lime_real(image_bytes, label)
