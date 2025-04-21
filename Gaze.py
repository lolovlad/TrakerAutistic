import cv2
import mediapipe as mp
import time
import math
import numpy as np
from GazeEstimation.GazeEstimation import GazeEstimation
from AllGaze.pygaze import PyGaze, PyGazeRenderer


mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils


LEFT_EYEBROW = [474, 475, 476, 477, 473]
RIGHT_EYEBROW = [469, 470, 471, 472, 468]
cap = cv2.VideoCapture(0)
gaze_eye_model = GazeEstimation("weight_file")

prev_left_eye = None
prev_right_eye = None

pg = PyGaze()
pgren = PyGazeRenderer()

with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_landmarks=True) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        orig_frame = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

        results = face_mesh.process(orig_frame)

        if results.multi_face_landmarks:
            gaze_result = pg.predict(orig_frame)
            for face in gaze_result:
                pgren.render(frame, face,draw_face_landmarks=True, draw_3dface_model=True,
                             draw_head_pose=True, draw_gaze_vector=True)

            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = orig_frame.shape

                eye_samples = gaze_eye_model.get_eyes(gray, face_landmarks.landmark)
                eye_preds = [gaze_eye_model.predict(i) for i in eye_samples]
                left_eyes = list(filter(lambda x: x.eye_sample.is_left, eye_preds))
                right_eyes = list(filter(lambda x: not x.eye_sample.is_left, eye_preds))

                prev_left_eye, prev_right_eye = gaze_eye_model.draw_landmarks(frame,
                                                                     (255, 0, 0),
                                                                     left_eyes[0],
                                                                     right_eyes[0],
                                                                     prev_left_eye,
                                                                     prev_right_eye)

        cv2.imshow("Iris Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break