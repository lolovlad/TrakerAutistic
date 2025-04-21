from pydantic import BaseModel
from .EyeSample import EyeSample


class EyePrediction(BaseModel):
    eye_sample: EyeSample
    landmarks: object
    gaze: object
