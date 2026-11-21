# Publicar el proyecto

## GitHub

1. Abre esta carpeta como repositorio local en GitHub Desktop. Si partes de una descarga sin historial, usa la opción de crear un repositorio local en esa misma carpeta.
2. Revisa los archivos que se van a incluir. Deben estar los dos notebooks, los archivos Python, el modelo, el README, las guías, los requisitos y la configuración de Streamlit. El archivo `.gitignore` excluye las imágenes originales, el ZIP, los entornos virtuales y los archivos de configuración con secretos.
3. Crea un commit en la rama principal. Pulsa **Publish repository**, elige el nombre y la cuenta u organización propietaria. Si quieres que cualquier persona pueda acceder, configura el repositorio como público.
4. Abre el repositorio en GitHub y comprueba que se vean el README, los dos notebooks y el archivo del modelo. [Guía oficial de GitHub Desktop](https://docs.github.com/en/desktop/adding-and-cloning-repositories/adding-an-existing-project-to-github-using-github-desktop)

## Streamlit Community Cloud

1. Entra en [Streamlit Community Cloud](https://share.streamlit.io/) y conecta la cuenta de GitHub que tiene acceso al repositorio.
2. Selecciona **Create app** y luego **Yup, I have an app**. Elige el repositorio, su rama principal y `app.py` como archivo de entrada.
3. En **Advanced settings**, selecciona **Python 3.13** y guarda. Community Cloud instala las dependencias de `requirements.txt`.
4. Pulsa **Deploy** y espera a que cargue la aplicación. Carga una imagen de prueba y confirma que se muestran la categoría, las dos puntuaciones y Grad-CAM.
5. Copia la URL real que termina en `.streamlit.app`. Sustituye el estado pendiente en **Aplicación en línea** del README y añade el enlace al campo **Website** del repositorio de GitHub. Si el despliegue debe ser público, verifica el acceso desde una ventana privada. [Guía oficial de despliegue](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
