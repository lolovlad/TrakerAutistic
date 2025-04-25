import mediapipe as mp
import cv2 as cv
import os

# Инициализация Mediapipe для Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_face_detection = mp.solutions.face_detection

os.chdir('C:/Users/rom17/Documents')

image = None
image_rgb = cv.cvtColor(image, cv.BGR2RGB)

height,weight = image.shape[0], image.shape[1]


    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
    # Преобразуем изображение из BGR в RGB
        results = face_detection.process(cv.cvtColor(image, cv.COLOR_BGR2RGB))
        for detection in results.detections:
    # Получаем координаты глаз
            left_eye = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.LEFT_EYE)
            right_eye = mp_face_detection.get_key_point(detection, mp_face_detection.FaceKeyPoint.RIGHT_EYE)

            # Преобразуем координаты в целые числа
            left_eye_x, left_eye_y = int(left_eye.x * image.shape[1]), int(left_eye.y * image.shape[0])
            right_eye_x, right_eye_y = int(right_eye.x * image.shape[1]), int(right_eye.y * image.shape[0])

            # Определяем границы для вырезания
            eye_width = 100  # ширина области для глаз (можно подкорректировать)
            eye_height = 50  # высота области для глаз (можно подкорректировать)

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

            rweight1, rheight1 = eyer_image.shape[1], eyer_image.shape[0]
            eyer_image = cv.resize(eyer_image,(int(rweight1*2),int(rheight1*2)))
            lweight, lheight = eyel_image.shape[1], eyel_image.shape[0]
            eyel_image = cv.resize(eyel_image,(int(lweight*2),int(lheight*2)))

            
            # Отображаем новое изображение с глазами
            cv.imshow('left eye', eyel_image)
            cv.imshow('right eye', eyer_image)
