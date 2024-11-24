import streamlit as st
import cv2
import mediapipe as mp
import google.generativeai as genai
from PIL import Image
import numpy as np
import io
import os
from datetime import datetime

# Configuración de la página de Streamlit
st.set_page_config(page_title="Detector de Gestos con IA", layout="wide")


# Configuración de Gemini API
def initialize_gemini():
    genai.configure(api_key="AIzaSyBeDEapDdHQN6OWRIuk5ufr-zarQru9XeU")
    model = genai.GenerativeModel('gemini-pro-vision')
    return model


# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)


def process_image(image):
    # Convertir la imagen a RGB si es necesario
    if len(image.shape) == 2:  # Si es imagen en escala de grises
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # Si tiene canal alpha
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # Procesar la imagen con MediaPipe
    results = hands.process(image)

    # Crear una copia de la imagen para dibujar
    annotated_image = image.copy()

    # Dibujar los puntos de referencia si se detectan manos
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    return annotated_image, results.multi_hand_landmarks


def analyze_gesture(image, landmarks):
    if not landmarks:
        return "No se detectaron manos en la imagen."

    # Convertir la imagen para Gemini
    pil_image = Image.fromarray(image)

    # Preparar el prompt para Gemini
    prompt = """
    Analiza esta imagen que contiene una mano detectada y responde:
    1. ¿Qué gesto está haciendo la mano?
    2. ¿Cuál podría ser el significado de este gesto en la comunicación no verbal?
    3. ¿Hay alguna característica particular notable en la posición de los dedos?
    Por favor, proporciona una respuesta concisa y clara.
    """

    try:
        model = initialize_gemini()
        response = model.generate_content([prompt, pil_image])
        return response.text
    except Exception as e:
        return f"Error al analizar la imagen con Gemini: {str(e)}"


def main():
    st.title("🖐️ Detector de Gestos con IA")
    st.write("Sube una imagen para detectar y analizar gestos de manos")

    # Subida de imagen
    uploaded_file = st.file_uploader("Elige una imagen", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Convertir la imagen subida a formato numpy
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Crear columnas para mostrar las imágenes
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagen Original")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

        # Procesar la imagen
        processed_image, landmarks = process_image(image)

        with col2:
            st.subheader("Imagen Procesada")
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), use_column_width=True)

        # Análisis con Gemini
        if st.button("Analizar Gesto"):
            with st.spinner("Analizando el gesto..."):
                analysis = analyze_gesture(processed_image, landmarks)
                st.markdown("### Análisis del Gesto")
                st.write(analysis)


if __name__ == "__main__":
    main()