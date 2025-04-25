import cv2,os
import numpy as np
import mediapipe as mp


# Инициализация Mediapipe для Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Загружаем изображение
image = cv2.imread('firts1.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Используем Mediapipe для обработки изображения
with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
    # Преобразуем изображение из BGR в RGB
    
    results = face_mesh.process(image_rgb)

    # Если обнаружены лица
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Рисуем сетку
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
            )

# Отображаем изображение с сеткой
cv2.imshow('Face Mesh', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
