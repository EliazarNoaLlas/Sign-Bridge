import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuración del modelo Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Diccionario para gestos básicos
GESTOS = {
    "0": "Hola",
    "1": "Adiós",
    "2": "Gracias",
}

def detectar_gesto(landmarks):
    """
    Detecta gestos básicos en función de las posiciones de los puntos clave de la mano.
    """
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

    # Ejemplo de lógica simplificada para detectar gestos
    if index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y:
        return "0"  # Gesto 'Hola'qq
    elif index_tip.y > thumb_tip.y and middle_tip.y > thumb_tip.y:
        return "1"  # Gesto 'Adiós'
    elif index_tip.y < middle_tip.y:
        return "2"  # Gesto 'Gracias'
    return None

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Convertir la imagen de BGR a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar el fotograma con MediaPipe Hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Detectar gesto
            landmarks = hand_landmarks.landmark
            gesto_id = detectar_gesto(landmarks)
            if gesto_id:
                texto_gesto = GESTOS.get(gesto_id, "Desconocido")
                cv2.putText(frame, f"Gesto: {texto_gesto}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar el video procesado
    cv2.imshow('Detección de Gestos de Lenguaje de Señas', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()