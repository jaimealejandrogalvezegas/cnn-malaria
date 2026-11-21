"""Inferencia del modelo de la fase 1 a partir de una imagen local."""

from io import BytesIO
from pathlib import Path

import numpy as np
import tensorflow as tf
from matplotlib import colormaps
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent / "models" / "modelo_malaria_cnn.keras"
IMAGE_SIZE = (48, 48)
THRESHOLD = 0.5
MAX_UPLOAD_MB = 10
MAX_PIXELS = 12_000_000
FORMATS = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}


def decode_upload(data: bytes, filename: str) -> tf.Tensor:
    """Valida el archivo y decodifica sus píxeles RGB de 8 bits."""
    if not data or len(data) > MAX_UPLOAD_MB * 1024**2:
        raise ValueError(f"Selecciona una imagen de hasta {MAX_UPLOAD_MB} MB.")
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in FORMATS.values() or image.format != FORMATS.get(
                Path(filename).suffix.lower()
            ):
                raise ValueError("La extensión debe coincidir con el formato PNG o JPEG del archivo.")
            if getattr(image, "n_frames", 1) != 1:
                raise ValueError("Selecciona una imagen estática de un solo cuadro.")
            width, height = image.size
            if width * height > MAX_PIXELS:
                raise ValueError("La imagen supera el límite de 12 megapíxeles.")
            if min(width, height) * IMAGE_SIZE[0] < max(width, height):
                raise ValueError("La imagen es demasiado alargada para el tamaño de entrada del modelo.")
            image_format = image.format
            image.verify()
    except Image.DecompressionBombError as error:
        raise ValueError("La imagen supera el límite de 12 megapíxeles.") from error
    except (OSError, SyntaxError) as error:
        raise ValueError("El archivo está dañado o no se puede abrir como imagen.") from error

    decode = tf.io.decode_png if image_format == "PNG" else tf.io.decode_jpeg
    try:
        return decode(data, channels=3)
    except tf.errors.OpError as error:
        raise ValueError("No se pudieron leer los píxeles de la imagen.") from error


def prepare_image(rgb: tf.Tensor) -> tf.Tensor:
    """Replica resize_cell del notebook y añade la dimensión de lote."""
    resized = tf.image.resize_with_pad(rgb, *IMAGE_SIZE, antialias=True)
    return resized[None, ...]


def load_model() -> tf.keras.Model:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    if (
        model.input_shape != (None, *IMAGE_SIZE, 3)
        or model.output_shape != (None, 1)
        or tf.keras.activations.serialize(model.layers[-1].activation) != "sigmoid"
    ):
        raise ValueError("El archivo no tiene la entrada y salida esperadas de la CNN.")
    return model


def parasitized_score(model: tf.keras.Model, batch: tf.Tensor) -> float:
    # Rescaling ya forma parte del modelo. training=False desactiva el aumento y Dropout.
    score = float(model(batch, training=False)[0, 0])
    if not 0 <= score <= 1:
        raise ValueError("El modelo devolvió una puntuación inválida.")
    return score


def preview_image(rgb: tf.Tensor) -> Image.Image:
    preview = Image.fromarray(rgb.numpy())
    preview.thumbnail((560, 560), Image.Resampling.LANCZOS)
    return preview


def grad_cam(model: tf.keras.Model, batch: tf.Tensor, positive: bool) -> np.ndarray:
    """Calcula el mapa Grad-CAM para la categoría predicha."""
    try:
        features = model.get_layer("relu_conv_3").output
        logits = model.get_layer("logit").output
        activation_model = tf.keras.Model(model.input, [features, logits])
    except (AttributeError, ValueError) as error:
        raise ValueError("La arquitectura guardada no tiene las capas previstas para Grad-CAM.") from error

    with tf.GradientTape() as tape:
        tape.watch(batch)
        activations, logits = activation_model(batch, training=False)
        score = logits[0, 0] if positive else -logits[0, 0]
    gradients = tape.gradient(score, activations)
    if gradients is None:
        raise ValueError("El modelo no devolvió gradientes para esta imagen.")

    weights = tf.reduce_mean(gradients[0], axis=(0, 1))
    heat = tf.nn.relu(tf.reduce_sum(activations[0] * weights, axis=-1))
    heat = tf.math.divide_no_nan(heat, tf.reduce_max(heat))
    return tf.image.resize(heat[..., None], IMAGE_SIZE)[..., 0].numpy()


def render_grad_cam(
    heat: np.ndarray, source_shape: tuple[int, int], preview: Image.Image
) -> tuple[Image.Image, Image.Image]:
    """Retira el relleno de la entrada y alinea el mapa con la vista original."""
    # Igual que resize_with_pad, el origen se redondea antes de convertir a enteros.
    source = np.asarray(source_shape, dtype=np.float32)
    target = np.asarray(IMAGE_SIZE, dtype=np.float32)
    resized = source / np.max(source / target)
    top, left = np.floor((target - resized) / 2).astype(int)
    height, width = np.floor(resized).astype(int)
    cropped = heat[top:top + height, left:left + width]
    if not cropped.size or not np.isfinite(cropped).all() or cropped.max() <= 0:
        raise ValueError("No hubo activación positiva en la región de la imagen para esta clase.")

    resized_heat = Image.fromarray(cropped).resize(preview.size, Image.Resampling.BILINEAR)
    intensity = np.clip(np.asarray(resized_heat), 0, 1)
    color = colormaps["magma"](intensity, bytes=True)[..., :3]
    alpha = (0.65 * intensity)[..., None]
    blend = (1 - alpha) * np.asarray(preview) + alpha * color
    return Image.fromarray(color), Image.fromarray(np.rint(blend).astype(np.uint8))

