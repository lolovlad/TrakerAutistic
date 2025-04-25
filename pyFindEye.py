
import cv2
import os
import mediapipe as mp

# Инициализация Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Указываем путь к изображению
os.chdir('C:/Users/rom17/Documents')
image = cv2.imread('firts1.jpg')

with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    # Преобразуем изображение из BGR в RGB
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Если обнаружены лица
    if results.detections:
        for detection in results.detections:
            # Получаем координаты глаз
            left_eye = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
            right_eye = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)

            # Преобразуем координаты в целые числа
            left_eye_x, left_eye_y = int(left_eye.x * image.shape[1]), int(left_eye.y * image.shape[0])
            right_eye_x, right_eye_y = int(right_eye.x * image.shape[1]), int(right_eye.y * image.shape[0])

            # Определяем границы для вырезания
            eye_width = 200  # ширина области для глаз (можно подкорректировать)
            eye_height = 100  # высота области для глаз (можно подкорректировать)

            # Урезаем область вокруг глаз
            # Убедимся, что координаты не выходят за границы изображения
            left_x1 = max(0, left_eye_x - eye_width // 2)
            left_x2 = min(image.shape[1], left_eye_x + eye_width // 2)
            left_y1 = max(0, left_eye_y - eye_height // 2)
            left_y2 = min(image.shape[0], left_eye_y + eye_height // 2)

            right_x1 = max(0, right_eye_x - eye_width // 2)
            right_x2 = min(image.shape[1], right_eye_x + eye_width // 2)
            right_y1 = max(0, right_eye_y - eye_height // 2)
            right_y2 = min(image.shape[0], right_eye_y + eye_height // 2)

            # Создаем новое изображение с глазной областью
            eyel_image = image[left_y1:left_y2, left_x1:left_x2]
            eyer_image = image[right_y1:right_y2, right_x1:right_x2]
            #eyes_image = cv2.hconcat([eyes_image, image[right_y1:right_y2, right_x1:right_x2]])

            # Отображаем новое изображение с глазами
            cv2.imshow('left eye', eyel_image)
            cv2.imshow('right eye', eyer_image)
            cv2.waitKey(0)

cv2.destroyAllWindows()

'''
import os
import cv2 as cv
import numpy as np
import mediapipe as mp

os.chdir('C:/Users/rom17/Documents')

# Инициализация Mediapipe для Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Загружаем изображение
image = cv.imread('firts1.jpg')  # Обратите внимание на название файла
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
height,weight = image.shape[0], image.shape[1]

# Используем Mediapipe для обработки изображения
with mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
    results = face_mesh.process(image_rgb)

    # Если обнаружены лица
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Индексы для левого и правого глаза
            left_eye_indices = [362, 385, 387, 263, 373, 380]  # Индексы для левого глаза
            right_eye_indices = [33, 160, 158, 133, 153, 144]  # Индексы для правого глаза
            
            # Собираем точки для левого глаза
            left_eye_points = np.array(
                [(int(face_landmarks.landmark[i].x * weight),
                  int(face_landmarks.landmark[i].y * height)) for i in left_eye_indices],
                dtype=np.int32
            )

            # Собираем точки для правого глаза
            right_eye_points = np.array(
                [(int(face_landmarks.landmark[i].x * weight),
                  int(face_landmarks.landmark[i].y * height)) for i in right_eye_indices],
                dtype=np.int32
            )

            # Рисуем контуры глаз на оригинальном изображении
            cv.polylines(image, [left_eye_points], isClosed=True, color=(255, 0, 0), thickness=2)  # Левый глаз
            cv.polylines(image, [right_eye_points], isClosed=True, color=(255, 0, 0), thickness=2)  # Правый глаз

    # Сохранение изображения с выделением глаз
    cv.imwrite('./highlighted_eyes.jpg', image)

    image = cv.resize(image,(int(weight*0.5),int(height*0.5)))
    
    # Отображение результата
    cv.imshow('Highlighted Eyes', image)
    cv.waitKey(0)
    cv.destroyAllWindows()
'''
