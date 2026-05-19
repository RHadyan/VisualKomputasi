import numpy as np
import io
import base64
from PIL import Image, ImageOps

_use_dummy = True


def generate_gradcam_dummy(image_bytes: bytes) -> str:
    """
    Generate a dummy Grad-CAM heatmap for testing.
    Returns a base64-encoded PNG image of the heatmap overlay.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)

    # Create a fake heatmap (gaussian blob in center)
    x = np.linspace(-1, 1, 224)
    y = np.linspace(-1, 1, 224)
    xx, yy = np.meshgrid(x, y)

    # Random offset for variety
    seed = int(np.array(img).sum()) % 100
    offset_x = (seed % 10 - 5) / 10
    offset_y = (seed // 10 - 5) / 10

    heatmap = np.exp(-((xx - offset_x) ** 2 + (yy - offset_y) ** 2) / 0.5)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # Apply colormap (red-yellow)
    heatmap_colored = np.zeros((224, 224, 3), dtype=np.float32)
    heatmap_colored[:, :, 0] = heatmap  # Red channel
    heatmap_colored[:, :, 1] = heatmap * 0.5  # Green channel (for yellow tint)
    heatmap_colored[:, :, 2] = 0  # Blue channel

    # Overlay on original image
    alpha = 0.4
    overlay = (1 - alpha) * (img_array / 255.0) + alpha * heatmap_colored
    overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)

    # Convert to base64
    overlay_img = Image.fromarray(overlay)
    buffer = io.BytesIO()
    overlay_img.save(buffer, format="PNG")
    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_gradcam_real(image_bytes: bytes) -> str:
    import tensorflow as tf
    from .model import _model

    if _model is None:
        return generate_gradcam_dummy(image_bytes)

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img.thumbnail((224, 224), Image.LANCZOS)
    delta_w = 224 - img.size[0]
    delta_h = 224 - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    img = ImageOps.expand(img, padding, fill=(255, 255, 255))
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

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_gradcam(image_bytes: bytes) -> str:
    """Generate Grad-CAM heatmap. Uses real model if available, otherwise dummy."""
    from .model import _use_dummy

    if _use_dummy:
        return generate_gradcam_dummy(image_bytes)
    else:
        return generate_gradcam_real(image_bytes)
