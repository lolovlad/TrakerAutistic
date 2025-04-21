from pydantic import BaseModel


class EyeSample(BaseModel):
    origin_img: object
    img: object
    is_left: bool
    transform_matrix: object
    estimated_radius: float
