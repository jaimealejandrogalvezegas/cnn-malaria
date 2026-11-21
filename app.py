"""Interfaz local de clasificación experimental de células sanguíneas."""

import logging

import streamlit as st
import tensorflow as tf

from malaria_inference import (
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

LOGGER = logging.getLogger(__name__)
TITLE = "Sistema inteligente de clasificación de células sanguíneas mediante CNN."
CLASS_NAMES = ("Uninfected", "Parasitized")

st.set_page_config(page_title="Clasificación de células · CNN", layout="wide")
st.html("""
<style>
  .stApp {
    background: radial-gradient(ellipse at 88% 0%, #e0f0ed 0%, transparent 32rem), #f3f7fa;
  }
  .block-container { max-width: 70rem; padding-top: 3.25rem; padding-bottom: 2.5rem; }
  .st-key-encabezado { border-top: 4px solid #08786d; padding-top: 1.1rem; margin-bottom: 1.1rem; }
  .st-key-encabezado h1 { max-width: 33ch; letter-spacing: -.045em; line-height: 1.05; }
  .st-key-encabezado p { max-width: 65ch; font-size: 1.05rem; }
  .st-key-carga, .st-key-resultado, .st-key-gradcam { background: #ffffff; }
  .st-key-carga, .st-key-resultado { min-height: 19rem; }
  .st-key-carga img, .st-key-gradcam img { border-radius: .6rem; }
  .st-key-resultado [data-testid="stMetricValue"] { font-size: clamp(1.45rem, 3vw, 2rem); }
  @media (max-width: 640px) {
    .block-container { padding-top: 2.5rem; }
    .st-key-carga, .st-key-resultado { min-height: 0; }
  }
</style>
""")


@st.cache_resource
def model_resource():
    return load_model()


def clear_analysis():
    st.session_state.pop("analysis", None)


def analyze(rgb, preview):
    batch = prepare_image(rgb)
    model = model_resource()
    score = parasitized_score(model, batch)

    try:
        heat = grad_cam(model, batch, positive=score >= THRESHOLD)
        views = render_grad_cam(heat, tuple(rgb.shape[:2]), preview)
    except (ValueError, tf.errors.OpError) as error:
        LOGGER.info("Grad-CAM no disponible", exc_info=True)
        return score, None, str(error) if isinstance(error, ValueError) else (
            "No se pudo calcular el mapa para esta imagen."
        )
    return score, views, None


def show_result(score):
    label = CLASS_NAMES[score >= THRESHOLD]
    st.subheader(label)
    description = (
        "La puntuación de Parasitized alcanzó el umbral de 50 %."
        if label == "Parasitized"
        else "La puntuación de Parasitized quedó por debajo del umbral de 50 %."
    )
    st.caption(description)
    for name, probability in (("Parasitized", score), ("Uninfected", 1 - score)):
        st.metric(name, f"{probability:.2%}".replace(".", ",").replace("%", " %"))
        st.progress(probability)


with st.container(key="encabezado"):
    st.title(TITLE)
    st.markdown("Carga una imagen microscópica de una célula sanguínea y consulta la clasificación de la CNN.")
    st.caption("Prototipo universitario para análisis experimental. Las puntuaciones no representan certeza clínica ni constituyen un diagnóstico.")

upload_column, result_column = st.columns(2, gap="medium")
preview = None
with upload_column.container(border=True, key="carga"):
    st.header("Imagen", anchor=False)
    uploaded = st.file_uploader(
        "Selecciona una imagen microscópica",
        type=["png", "jpg", "jpeg"],
        key="microscopia",
        label_visibility="collapsed",
        on_change=clear_analysis,
    )
    st.caption(f"PNG, JPG o JPEG · Hasta {MAX_UPLOAD_MB} MB y 12 megapíxeles")
    if uploaded is not None:
        try:
            rgb = decode_upload(uploaded.getvalue(), uploaded.name)
            preview = preview_image(rgb)
        except ValueError as error:
            clear_analysis()
            st.error(str(error))

    if st.button("Analizar imagen", type="primary", disabled=preview is None, width="stretch"):
        clear_analysis()
        with st.spinner("Analizando la imagen…"):
            try:
                st.session_state.analysis = analyze(rgb, preview)
            except (OSError, ValueError, TypeError, AttributeError, tf.errors.OpError):
                LOGGER.exception("No se pudo completar la clasificación")
                st.error("No se pudo completar la clasificación con el modelo guardado.")

    if preview is not None:
        height, width = rgb.shape[:2]
        st.image(preview, caption=f"Original · {width} × {height} píxeles", width=320)

result = st.session_state.get("analysis")
with result_column.container(border=True, key="resultado"):
    st.header("Resultado", anchor=False)
    if result is None:
        st.info("Aquí verás la categoría y las puntuaciones después de analizar la imagen.")
    else:
        show_result(result[0])

if result is not None:
    _, views, explanation = result
    with st.container(border=True, key="gradcam"):
        st.header("Grad-CAM", anchor=False)
        if views is None:
            st.info(explanation)
        else:
            st.caption("Las zonas más claras del mapa aportaron más a la categoría predicha en esta imagen.")
            for column, image, caption in zip(
                st.columns(3, gap="medium"), (preview, *views),
                ("Imagen original", "Mapa de calor", "Superposición"),
            ):
                column.image(image, caption=caption, width="stretch")

