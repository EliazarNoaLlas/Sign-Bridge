import cv2
import mediapipe as mp

# Inicializar MediaPipe Hands y herramientas de dibujo
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuración del modelo Hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Procesar flujo de video en lugar de imágenes estáticas.
    max_num_hands=2,  # Detectar hasta 2 manos.
    min_detection_confidence=0.5,  # Confianza mínima para detección inicial.
    min_tracking_confidence=0.5   # Confianza mínima para rastrear puntos clave.
)

# Captura de video de la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    # Convertir la imagen de BGR (formato de OpenCV) a RGB (formato requerido por MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Procesar el marco con MediaPipe Hands
    result = hands.process(rgb_frame)

    # Dibujar puntos clave y conexiones
    if result.multi_hand_landmarks:  # Si hay manos detectadas
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Imprimir las coordenadas de los puntos clave
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)  # Coordenadas escaladas al tamaño del marco.
                print(f"Punto {id}: ({cx}, {cy})")

    # Mostrar el video procesado
    cv2.imshow('MediaPipe Hands', frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()