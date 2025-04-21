from pathlib import Path
from .models.eyenet import EyeNet
from .model_data import EyeSample, EyePrediction
from typing import Optional
from .util.gaze import draw_gaze

import torch
import numpy as np
import cv2
import os
from loguru import logger


class GazeEstimation:
    def __download_model(self, path: str | Path):
        output_dir = Path(path).expanduser()
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = Path(output_dir, 'checkpoint.pt')
        if not os.path.exists(output_path):
            torch.hub.download_url_to_file(
                'https://github.com/lolovlad/TrakerAutistic/releases/download/model/checkpoint.pt',
                str(output_path))
        else:
            logger.debug('The pretrained model {} already exists.', output_path)
        return output_path

    def __init__(self, path_model: str | Path):
        torch.backends.cudnn.enabled = True
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__checkpoint = torch.load(self.__download_model(path_model), map_location=self.__device, weights_only=False)
        self.__model = EyeNet(nstack=self.__checkpoint['nstack'],
                              nfeatures=self.__checkpoint['nfeatures'],
                              nlandmarks=self.__checkpoint['nlandmarks']).to(self.__device)
        self.__model.load_state_dict(self.__checkpoint['model_state_dict'])
        self.__weight = 160
        self.__height = 96
        self.__LEFT_EYE_LANDMARKS = [463, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374,
                                     380, 381, 382, 362]

        self.__RIGHT_EYE_LANDMARKS = [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145,
                                      144, 163, 7]

    def predict(self, eye: EyeSample) -> EyePrediction:
        with torch.no_grad():
            x = torch.tensor([eye.img], dtype=torch.float32).to(self.__device)
            _, landmarks, gaze = self.__model.forward(x)
            landmarks = np.asarray(landmarks.cpu().numpy()[0])
            gaze = np.asarray(gaze.cpu().numpy()[0])

            landmarks = landmarks * np.array([self.__height / 48, self.__weight / 80])

            temp = np.zeros((landmarks.shape[0], 3))
            if eye.is_left:
                temp[:, 0] = self.__weight - landmarks[:, 1]
            else:
                temp[:, 0] = landmarks[:, 1]
            temp[:, 1] = landmarks[:, 0]
            temp[:, 2] = 1.0
            landmarks = temp
            landmarks = np.asarray(np.matmul(landmarks, eye.transform_matrix.T))[:, :2]
            return EyePrediction(eye_sample=eye, landmarks=landmarks, gaze=gaze)

    def bounding_rectangle(self, points):
        if not points:
            raise ValueError("Список точек не должен быть пустым")

        x_min = min(x for x, y in points)
        y_min = min(y for x, y in points)
        x_max = max(x for x, y in points)
        y_max = max(y for x, y in points)

        return x_min, y_min, x_max, y_max

    def get_point_landmarks(self, landmarks, points, scale):
        get_points = []
        for i in points:
            get_points.append((landmarks[i].x * scale[1], landmarks[i].y * scale[0]))
        return get_points

    def get_eyes(self, frame, landmarks):
        eyes = []
        left_eye = self.get_point_landmarks(landmarks, self.__LEFT_EYE_LANDMARKS, frame.shape)
        right_eye = self.get_point_landmarks(landmarks, self.__RIGHT_EYE_LANDMARKS, frame.shape)
        for points, is_left in [(left_eye, True), (right_eye, False)]:
            x1, y1, x2, y2 = self.bounding_rectangle(points)
            eye_width = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
            if eye_width == 0.0:
                return eyes

            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)

            # center image on middle of eye
            translate_mat = np.asmatrix(np.eye(3))
            translate_mat[:2, 2] = [[-cx], [-cy]]
            inv_translate_mat = np.asmatrix(np.eye(3))
            inv_translate_mat[:2, 2] = -translate_mat[:2, 2]

            # Scale
            scale = self.__weight / eye_width
            scale_mat = np.asmatrix(np.eye(3))
            scale_mat[0, 0] = scale_mat[1, 1] = scale
            inv_scale = 1.0 / scale
            inv_scale_mat = np.asmatrix(np.eye(3))
            inv_scale_mat[0, 0] = inv_scale_mat[1, 1] = inv_scale

            estimated_radius = 0.5 * eye_width * scale

            # center image
            center_mat = np.asmatrix(np.eye(3))
            center_mat[:2, 2] = [[0.5 * self.__weight], [0.5 * self.__height]]
            inv_center_mat = np.asmatrix(np.eye(3))
            inv_center_mat[:2, 2] = -center_mat[:2, 2]

            # Get rotated and scaled, and segmented image
            transform_mat = center_mat * scale_mat * translate_mat
            inv_transform_mat = (inv_translate_mat * inv_scale_mat * inv_center_mat)

            eye_image = cv2.warpAffine(frame, transform_mat[:2, :], (self.__weight, self.__height))
            eye_image = cv2.equalizeHist(eye_image)

            if is_left:
                eye_image = np.fliplr(eye_image)
                cv2.imshow('left eye image', eye_image)

            else:
                cv2.imshow('right eye image', eye_image)
            eyes.append(EyeSample(origin_img=frame.copy(),
                                  img=eye_image,
                                  transform_matrix=inv_transform_mat,
                                  is_left=is_left,
                                  estimated_radius=estimated_radius))
        return eyes

    def smooth_eye_landmarks(self, eye: EyePrediction, prev_eye: Optional[EyePrediction], smoothing=0.2, gaze_smoothing=0.4):
        if prev_eye is None:
            return eye
        return EyePrediction(
            eye_sample=eye.eye_sample,
            landmarks=smoothing * prev_eye.landmarks + (1 - smoothing) * eye.landmarks,
            gaze=gaze_smoothing * prev_eye.gaze + (1 - gaze_smoothing) * eye.gaze)

    def draw_landmarks(self, frame, color, left_eye, right_eye, prev_left_eye, prev_right_eye):
        if left_eye:
            left_eye = self.smooth_eye_landmarks(left_eye, prev_left_eye, smoothing=0.1)
        if right_eye:
            right_eye = self.smooth_eye_landmarks(right_eye, prev_right_eye, smoothing=0.1)

        for ep in [left_eye, right_eye]:
            for (x, y) in ep.landmarks[16:33]:
                cv2.circle(frame,
                           (int(round(x)), int(round(y))), 1, color, -1, lineType=cv2.LINE_AA)
            gaze = ep.gaze.copy()
            if ep.eye_sample.is_left:
                gaze[1] = -gaze[1]
            draw_gaze(frame, ep.landmarks[-2], gaze, length=60.0, thickness=2)
        return left_eye, right_eye