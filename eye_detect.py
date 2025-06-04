import cv2
import mediapipe as mp
import math
import time

# ---------- Funciones auxiliares ----------
def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_ear(eye):
    A = euclidean(eye[1], eye[5])
    B = euclidean(eye[2], eye[4])
    C = euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def calculate_mar(top_lip, bottom_lip, left_mouth, right_mouth):
    A = euclidean(top_lip, bottom_lip)
    C = euclidean(left_mouth, right_mouth)
    return A / C

# ---------- Índices ----------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
TOP_LIP = 13
BOTTOM_LIP = 14
LEFT_MOUTH = 78
RIGHT_MOUTH = 308

# ---------- Umbrales ----------
EAR_THRESHOLD = 0.18
MAR_THRESHOLD = 0.5
SLEEP_DURATION = 2.0
YAWN_DURATION = 1.5

# ---------- Inicialización ----------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
cap = cv2.VideoCapture(0)

closed_eyes_start_time = None
yawn_start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Extraer puntos de ojos
            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]

            ear_left = calculate_ear(left_eye)
            ear_right = calculate_ear(right_eye)
            ear_avg = (ear_left + ear_right) / 2.0

            # Extraer puntos de boca
            top_lip = (int(landmarks[TOP_LIP].x * w), int(landmarks[TOP_LIP].y * h))
            bottom_lip = (int(landmarks[BOTTOM_LIP].x * w), int(landmarks[BOTTOM_LIP].y * h))
            left_mouth = (int(landmarks[LEFT_MOUTH].x * w), int(landmarks[LEFT_MOUTH].y * h))
            right_mouth = (int(landmarks[RIGHT_MOUTH].x * w), int(landmarks[RIGHT_MOUTH].y * h))

            mar = calculate_mar(top_lip, bottom_lip, left_mouth, right_mouth)

            # --------- Detección de ojos cerrados ---------
            if ear_avg < EAR_THRESHOLD:
                if closed_eyes_start_time is None:
                    closed_eyes_start_time = time.time()
                elif time.time() - closed_eyes_start_time >= SLEEP_DURATION:
                    cv2.putText(frame, "ADVERTENCIA: Somnolencia", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
            else:
                closed_eyes_start_time = None

            # --------- Detección de bostezo ---------
            if mar > MAR_THRESHOLD:
                if yawn_start_time is None:
                    yawn_start_time = time.time()
                elif time.time() - yawn_start_time >= YAWN_DURATION:
                    cv2.putText(frame, "BOSTEZO DETECTADO", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 165, 255), 3)
            else:
                yawn_start_time = None

            # Mostrar métricas (opcional)
            cv2.putText(frame, f'EAR: {ear_avg:.2f} MAR: {mar:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Detección de Somnolencia y Bostezo", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
