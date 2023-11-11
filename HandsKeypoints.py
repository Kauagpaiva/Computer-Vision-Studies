import cv2
import mediapipe as mp

# Inicialize o módulo Hands do MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Inicialize o módulo Drawing da MediaPipe para desenhar os keypoints
mp_drawing = mp.solutions.drawing_utils

# Inicialize a captura de vídeo da webcam
cap = cv2.VideoCapture(0)  # Use 0 para a câmera padrão

while True:
    # Capture o quadro da webcam
    ret, frame = cap.read()

    # Converta o quadro para escala de cinza
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Converta a imagem em escala de cinza para uma imagem colorida
    frame_rgb = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    # Detecte as mãos no quadro
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            # Desenhe os keypoints da mão no quadro
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Exiba o quadro
    cv2.imshow('Hand Keypoints Detection', frame)

    # Se pressionar a tecla 'q', saia do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libere os recursos
cap.release()
cv2.destroyAllWindows()

