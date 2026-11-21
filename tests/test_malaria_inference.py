"""Pruebas del prototipo con el modelo y las predicciones de la fase 1."""

import csv
from io import BytesIO
from pathlib import Path
import unittest

import numpy as np
import tensorflow as tf
from PIL import Image

from malaria_inference import (
    IMAGE_SIZE,
    MAX_UPLOAD_MB,
    THRESHOLD,
    decode_upload,
    grad_cam,
    load_model,
    parasitized_score,
    prepare_image,
    preview_image,
    render_grad_cam,
)

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "results" / "metrics" / "predicciones_prueba.csv"


def encoded_image(fmt, size=(14, 9), **options):
    buffer = BytesIO()
    Image.new("RGB", size, (128, 24, 72)).save(buffer, format=fmt, **options)
    return buffer.getvalue()


class UploadTests(unittest.TestCase):
    def test_png_jpg_and_jpeg_decode_as_rgb(self):
        for fmt, filename in [("PNG", "cell.PNG"), ("JPEG", "cell.jpg"), ("JPEG", "cell.jpeg")]:
            with self.subTest(filename=filename):
                rgb = decode_upload(encoded_image(fmt), filename)
                self.assertEqual(tuple(rgb.shape), (9, 14, 3))
                self.assertEqual(rgb.dtype, tf.uint8)
                prepared = prepare_image(rgb)
                self.assertEqual(tuple(prepared.shape), (1, 48, 48, 3))
                self.assertGreater(float(tf.reduce_max(prepared)), 1.0)

    def test_rejects_invalid_uploads(self):
        png = encoded_image("PNG")
        invalid = [
            ("empty.png", b"", "hasta 10 MB"),
            ("oversized.png", b"x" * (MAX_UPLOAD_MB * 1024**2 + 1), "hasta 10 MB"),
            ("corrupt.png", b"texto", "dañado"),
            ("truncated.png", png[:len(png) // 2], "dañado|leer los píxeles"),
            ("wrong_format.jpg", png, "extensión"),
            ("unsupported.png", encoded_image("GIF"), "formato"),
            ("too_many_pixels.png", encoded_image("PNG", (3500, 3500)), "12 megapíxeles"),
            ("too_narrow.png", encoded_image("PNG", (50, 1)), "alargada"),
            (
                "animated.png",
                encoded_image("PNG", save_all=True, append_images=[Image.new("RGB", (14, 9))]),
                "un solo cuadro",
            ),
        ]
        for name, payload, message in invalid:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, message):
                decode_upload(payload, name)


class GradCAMGeometryTests(unittest.TestCase):
    def test_padding_excluded_and_overlay_aligned(self):
        # Un mapa que solo activa el relleno no debe colorear la foto.
        for height, width in [(160, 104), (48, 1), (1, 48), (349, 381), (381, 349)]:
            with self.subTest(height=height, width=width):
                rgb = tf.ones((height, width, 3), dtype=tf.uint8)
                source_pixels = prepare_image(rgb).numpy()[0, :, :, 0] > 0
                preview = preview_image(rgb)
                with self.assertRaisesRegex(ValueError, "No hubo activación positiva"):
                    render_grad_cam((~source_pixels).astype(np.float32), (height, width), preview)

                heatmap, overlay = render_grad_cam(
                    source_pixels.astype(np.float32), (height, width), preview
                )
                self.assertEqual(heatmap.size, preview.size)
                self.assertEqual(overlay.size, preview.size)
                colors = np.asarray(heatmap).reshape(-1, 3)
                np.testing.assert_array_equal(colors, np.broadcast_to(colors[0], colors.shape))
                self.assertFalse(np.array_equal(np.asarray(overlay), np.asarray(preview)))

    def test_missing_gradcam_layer_reports_incompatibility(self):
        inputs = tf.keras.Input(shape=(48, 48, 3))
        model = tf.keras.Model(inputs, tf.keras.layers.GlobalAveragePooling2D()(inputs))
        with self.assertRaisesRegex(ValueError, "capas previstas para Grad-CAM"):
            grad_cam(model, tf.zeros((1, 48, 48, 3)), positive=True)


class SavedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()
        with PREDICTIONS.open(encoding="utf-8", newline="") as stream:
            cls.rows = list(csv.DictReader(stream))

    def test_gradcam_for_both_predicted_classes(self):
        for positive, target in [(False, 0.1), (True, 0.98)]:
            with self.subTest(positive=positive):
                row = min(self.rows, key=lambda item: abs(float(item["prob_cnn"]) - target))
                path = ROOT / row["ruta"]
                rgb = decode_upload(path.read_bytes(), path.name)
                batch = prepare_image(rgb)
                self.assertEqual(parasitized_score(self.model, batch) >= THRESHOLD, positive)
                heat = grad_cam(self.model, batch, positive)
                self.assertEqual(heat.shape, IMAGE_SIZE)
                self.assertTrue(np.isfinite(heat).all())
                self.assertGreater(float(heat.max()), 0)
                heatmap, overlay = render_grad_cam(heat, tuple(rgb.shape[:2]), preview_image(rgb))
                self.assertEqual(heatmap.size, overlay.size)

    def test_full_reserved_split_matches_notebook(self):
        self.assertEqual(len(self.rows), 4004)
        scores = []
        for start in range(0, len(self.rows), 64):
            prepared = []
            for row in self.rows[start:start + 64]:
                path = ROOT / row["ruta"]
                prepared.append(prepare_image(decode_upload(path.read_bytes(), path.name)))
            batch = tf.concat(prepared, axis=0)
            scores.extend(self.model(batch, training=False).numpy().ravel())
        scores = np.asarray(scores, dtype=np.float64)
        reference = np.asarray([float(row["prob_cnn"]) for row in self.rows])
        expected_labels = [int(row["pred_cnn"]) for row in self.rows]
        np.testing.assert_allclose(scores, reference, rtol=0, atol=2e-5)
        np.testing.assert_array_equal(scores >= THRESHOLD, expected_labels)
        print(f"\n{len(scores):,} imágenes verificadas; diferencia máxima: {abs(scores - reference).max():.3g}")

