import cv2
import mediapipe as mp
import math
import time

# Función para calcular la distancia euclidiana entre dos puntos
def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

# Detectar si el ojo está cerrado usando EAR (Eye Aspect Ratio)
def calculate_ear(eye_landmarks):
    A = euclidean(eye_landmarks[1], eye_landmarks[5])
    B = euclidean(eye_landmarks[2], eye_landmarks[4])
    C = euclidean(eye_landmarks[0], eye_landmarks[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Mapeo de los puntos para ojos de MediaPipe
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Inicialización de MediaPipe y cámara
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
cap = cv2.VideoCapture(0)

EAR_THRESHOLD = 0.18
DROWSY_TIME = 2  # segundos cerrando los ojos
start_drowsy = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb_frame)

    if result.multi_face_landmarks:
        for face in result.multi_face_landmarks:
            landmarks = face.landmark

            # Obtener coordenadas de los ojos
            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]

            # Calcular EAR de ambos ojos
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2

            # Mostrar los ojos
            for x, y in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Verificar si está "dormido"
            if avg_ear < EAR_THRESHOLD:
                if start_drowsy is None:
                    start_drowsy = time.time()
                elif time.time() - start_drowsy > DROWSY_TIME:
                    cv2.putText(frame, "DORMIDO", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)
            else:
                start_drowsy = None  # resetea si ojos están abiertos

            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Deteccion de Sueno", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()
