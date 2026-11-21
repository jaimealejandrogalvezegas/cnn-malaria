# Reproducir el proyecto

## Preparar el entorno

1. Descarga o clona el repositorio y abre una terminal en la carpeta donde están `README.md` y `app.py`.
2. Instala Python 3.13 de 64 bits. Crea un entorno virtual con `python -m venv .venv`. En Windows también puedes usar `py -3.13 -m venv .venv` para elegir esa versión.
3. Activa el entorno. En PowerShell usa `.\.venv\Scripts\Activate.ps1`. En macOS o Linux usa `source .venv/bin/activate`. Si PowerShell bloquea la activación, ejecuta los pasos siguientes sustituyendo `python` por `.\.venv\Scripts\python.exe`.
4. Instala las dependencias con `python -m pip install -r requirements.txt` para usar solo la aplicación. Si vas a ejecutar notebooks o repetir la evaluación, instala `python -m pip install -r requirements-notebook.txt`.

El flujo se verificó en Windows con CPU y Python 3.13. En otras plataformas se necesitan paquetes de TensorFlow compatibles con la arquitectura del equipo.

## Conseguir las imágenes para los notebooks

1. Descarga el [ZIP original del NIH](https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip). Si ya tienes las imágenes, usa esa copia.
2. Extrae las carpetas `Parasitized` y `Uninfected` dentro de `cell_images`, junto a `app.py`. Cada carpeta debe contener sus PNG directamente. Si el ZIP crea dos carpetas `cell_images` anidadas, usa la interior como carpeta de datos.
3. Conserva las tablas `patientid_cellmapping_parasitized.csv` y `patientid_cellmapping_uninfected.csv` en `data`. El entrenamiento las usa para dividir las imágenes por paciente. [Tablas oficiales del NIH](https://data.lhncbc.nlm.nih.gov/public/Malaria/index.html)

Las imágenes originales se obtienen de la fuente y quedan fuera de Git. La aplicación usa el modelo guardado y las imágenes que cargue cada persona.

## Ejecutar los notebooks

1. Inicia JupyterLab con `python -m jupyterlab` desde la raíz del proyecto.
2. Abre [PROYECTO_MALARIA_CNN.ipynb](PROYECTO_MALARIA_CNN.ipynb). Para reconstruir el modelo y la evaluación, selecciona el kernel del entorno virtual y usa **Run All Cells**. La ejecución escribe el modelo, los CSV de evaluación y las figuras en sus carpetas correspondientes. Hazla en una copia si quieres conservar los resultados publicados para compararlos.
3. Abre [GUIA_PROTOTIPO_MALARIA.ipynb](GUIA_PROTOTIPO_MALARIA.ipynb) y ejecuta todas sus celdas. La guía toma dos imágenes del conjunto de prueba y usa el mismo módulo de inferencia que Streamlit.

## Abrir y probar el prototipo

1. Ejecuta `python -m streamlit run app.py --server.address 127.0.0.1` desde la raíz.
2. Abre [http://127.0.0.1:8501](http://127.0.0.1:8501), carga una imagen y pulsa **Analizar imagen**.
3. Para repetir las pruebas con las imágenes originales y las dependencias de notebook instaladas, ejecuta `python -m unittest discover -s tests -v`. La suite verifica la carga de imágenes, Grad-CAM y la equivalencia con las predicciones guardadas para las 4 004 imágenes de prueba. Si acabas de reentrenar, la comparación usa los resultados escritos por esa ejecución.
